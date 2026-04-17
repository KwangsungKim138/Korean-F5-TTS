import os
import sys
import warnings
import re
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from omegaconf import OmegaConf
from hydra.utils import get_class
import jiwer
import whisper
import torchaudio # Use torchaudio instead of transformers for WavLM

# Filter warnings
warnings.filterwarnings("ignore", message="Plan failed with a cudnnException")
warnings.filterwarnings("ignore", category=UserWarning)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Add UTMOSv2 to path
utmos_path = os.path.join(os.path.dirname(__file__), "src/third_party/UTMOSv2")
sys.path.append(utmos_path)
try:
    import utmosv2
    print("✅ UTMOSv2 imported successfully.")
except ImportError as e:
    print(f"⚠️ Failed to import UTMOSv2: {e}")
    utmosv2 = None
except Exception as e:
    print(f"⚠️ Unexpected error importing UTMOSv2: {e}")
    utmosv2 = None

from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    infer_process,
    preprocess_ref_audio_text,
)

# N2gk+ normalization import
try:
    from f5_tts.train.datasets.normalization_n2gk import normalize_n2gk_plus
    print("✅ N2gkPlus normalization module loaded successfully.")
except ImportError:
    print("❌ Warning: 'normalization_n2gk.py' not found. Normalization will be skipped.")
    def normalize_n2gk_plus(text): return text

# --------------------------
# Configuration
# --------------------------

# 검증할 모델 방법론 목록
TARGET_MODES = [
    "grapheme",
    "phoneme",
    "salt_n",
    "salt_vcp",
]

# 검증할 스텝 범위 (단위: 1000 스텝)
# 예: 50부터 250까지 25 간격 -> 50, 75, 100, 125, 150, 175, 200, 225, 250
EVAL_STEPS = range(50, 351, 50)

# Mapping for legacy names
MODE_MAP = {
    "V+N+L-H": "efficient_allophone",
    "VCP": "allophone",
    "N": "n_only",
    "salt_n": "n_only",
    "salt_vcp": "allophone",
    "V": "i_only",
    "C": "c_only",
    "V+C": "i_and_c",
    "V+N": "i_and_n",
    "N+L": "nf",
    "V+N+L": "inf",
}

# Data Paths
DATA_ROOT = "data/KSS"
VAL_TXT_PATH = os.path.join(DATA_ROOT, "valid.txt")

if not os.path.exists(VAL_TXT_PATH) and os.path.exists("valid.txt"):
    VAL_TXT_PATH = "valid.txt"

# 테스트 결과와 섞이지 않도록 출력 폴더명을 변경
OUTPUT_BASE_DIR = "eval_results/KSS_1h_val"

# Inference Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCODER_NAME = "vocos"
TARGET_RMS = 0.1
CROSS_FADE_DURATION = 0.15
NFE_STEP = 32
CFG_STRENGTH = 2.0
SWAY_SAMPLING_COEF = -1.0
SPEED = 1.0
WHISPER_SIZE = "large-v3"

# --------------------------
# Helper 1: Reference Matching
# --------------------------

MIN_DURATION = 2.7
CHAR_DURATION_RATIO = 0.33

def parse_kss_line(line):
    parts = line.strip().split("|")
    if len(parts) < 5: return None
    try:
        # 4번째 컬럼(parts[3])을 pronunciation으로 가져옵니다.
        return {"path": parts[0], "text": parts[2], "duration": float(parts[4])}
    except ValueError: return None

def get_pure_char_count(text):
    return len(re.findall(r'[가-힣A-Za-z0-9]', text))

def get_target_punctuation(text):
    text = text.strip()
    if not text: return "."
    return text[-1] if text[-1] in ['.', '?', '!'] else "."

def is_valid_candidate(text):
    if ',' in text: return False
    if '.' in text[:-1]: return False
    return True

def build_reference_mapping(test_path):
    print("Building strict reverse reference mapping from validation set...")
    test_items = []
    
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = parse_kss_line(line)
            if item: 
                test_items.append(item)
    
    mapping = {}
    total_items = len(test_items)
    
    for i, t_item in enumerate(test_items):
        ref_index = total_items - 1 - i
        mapping[t_item['path']] = test_items[ref_index]
            
    return test_items, mapping

# --------------------------
# Helper 2: Evaluation Models
# --------------------------

def post_process_for_metric(text):
    text = re.sub(r'[^\w\s가-힣ㄱ-ㅎ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def load_vocoder_model():
    print(f"Loading Vocoder ({VOCODER_NAME})...")
    return load_vocoder(vocoder_name=VOCODER_NAME, is_local=False, device=DEVICE)

def load_whisper_model():
    print(f"Loading Whisper ({WHISPER_SIZE})...")
    return whisper.load_model(WHISPER_SIZE, device=DEVICE)

def load_utmos_model():
    print("Loading UTMOSv2 predictor...")
    if utmosv2 is None: return None
    try:
        model = utmosv2.create_model(pretrained=True)
        return model.eval().to(DEVICE)
    except Exception as e:
        print(f"Failed to load UTMOSv2: {e}")
        return None

def load_wavlm_sim_model():
    print(f"Loading WavLM (torchaudio bundle)...")
    try:
        bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
        model = bundle.get_model()
        return bundle, model.to(DEVICE).eval()
    except Exception as e:
        print(f"Failed to load WavLM (torchaudio): {e}")
        return None, None

def calculate_sim(bundle, model, ref_path, gen_path):
    try:
        ref_wav, sr1 = torchaudio.load(ref_path)
        gen_wav, sr2 = torchaudio.load(gen_path)
        
        target_sr = bundle.sample_rate
        if sr1 != target_sr:
            ref_wav = torchaudio.functional.resample(ref_wav, sr1, target_sr)
        if sr2 != target_sr:
            gen_wav = torchaudio.functional.resample(gen_wav, sr2, target_sr)
            
        ref_wav = ref_wav.to(DEVICE)
        gen_wav = gen_wav.to(DEVICE)

        with torch.no_grad():
            ref_out = model(ref_wav)[0] 
            gen_out = model(gen_wav)[0]
            
            ref_emb = ref_out.mean(dim=1)
            gen_emb = gen_out.mean(dim=1)
            
        return F.cosine_similarity(ref_emb, gen_emb, dim=-1).item()
    except Exception as e:
        print(f"SIM Error: {e}")
        return np.nan

# --------------------------
# Main Logic
# --------------------------

def run_evaluation():
    if not os.path.exists(VAL_TXT_PATH):
        print(f"Error: Dataset files not found at {VAL_TXT_PATH}.")
        return

    test_items, ref_mapping = build_reference_mapping(VAL_TXT_PATH)
    total_items = len(test_items)
    
    vocoder = None
    whisper_model = None
    utmos_predictor = None
    wavlm_bundle, wavlm_model = None, None
    
    print("Pre-processing reference audios...")
    ref_cache = {}
    
    final_summary = []

    # --------------------------
    # Evaluate Ground Truth
    # --------------------------
    gt_dir = os.path.join(OUTPUT_BASE_DIR, "GT")
    gt_wav_dir = os.path.join(gt_dir, "wavs")
    gt_details_path = os.path.join(gt_dir, "details.csv")
    os.makedirs(gt_wav_dir, exist_ok=True)

    gt_done = False
    if os.path.exists(gt_details_path):
        try:
            df = pd.read_csv(gt_details_path)
            if len(df) == total_items:
                gt_done = True
                print(f"[GT] Results found. Loading...")
                refs_gt = df['gt'].fillna("").tolist()
                hyps_gt = df['pred'].fillna("").tolist()
                refs_cer = [r.replace(" ", "") for r in refs_gt]
                hyps_cer = [h.replace(" ", "") for h in hyps_gt]
                final_summary.append({
                    "Model": "GroundTruth", "Step": "N/A",
                    "CER": min(1.0, jiwer.cer(refs_cer, hyps_cer)) if refs_cer else 0.0,
                    "WER": min(1.0, jiwer.wer(refs_gt, hyps_gt)) if refs_gt else 0.0,
                    "UTMOS": df['utmos'].mean(), "SIM": df['sim'].mean()
                })
        except: pass

    if not gt_done:
        print(f"\n==================================================")
        print(f"Evaluating Ground Truth (GT)")
        print(f"==================================================")
        
        print(f"[GT] Preparing audio files in {gt_wav_dir}...")
        for item in test_items:
            src_path = os.path.join(DATA_ROOT, item['path'])
            dst_path = os.path.join(gt_wav_dir, os.path.basename(item['path']))
            if not os.path.exists(dst_path):
                try:
                    os.symlink(os.path.abspath(src_path), dst_path)
                except OSError:
                    import shutil
                    shutil.copy(src_path, dst_path)

        if whisper_model is None: whisper_model = load_whisper_model()
        if utmos_predictor is None: utmos_predictor = load_utmos_model()
        if wavlm_model is None: wavlm_bundle, wavlm_model = load_wavlm_sim_model()

        gt_utmos_results = {}
        if utmos_predictor is not None:
            print(f"[GT] Running bulk UTMOS prediction...")
            try:
                bulk_preds = utmos_predictor.predict(input_dir=gt_wav_dir, device=DEVICE, verbose=False)
                for p in bulk_preds:
                    fname = os.path.basename(p['file_path'])
                    gt_utmos_results[fname] = p['predicted_mos']
            except Exception as e:
                print(f"GT UTMOS Error: {e}")

        gt_metrics = []
        for item in tqdm(test_items, desc="Evaluating GT"):
            test_fname = os.path.basename(item['path'])
            wav_path = os.path.join(gt_wav_dir, test_fname)
            
            try:
                asr_out = whisper_model.transcribe(wav_path, language='ko', temperature=0.0)
                pred_text = asr_out['text']
            except:
                print(f"Whisper Error: {e}")
                pred_text = ""

            gt_clean = post_process_for_metric(normalize_n2gk_plus(item['text']))
            pred_clean = post_process_for_metric(normalize_n2gk_plus(pred_text))
            gt_ns = gt_clean.replace(" ", "")
            pred_ns = pred_clean.replace(" ", "")

            cer = min(1.0, jiwer.cer(gt_ns, pred_ns)) if gt_ns else 1.0
            wer = min(1.0, jiwer.wer(gt_clean, pred_clean)) if gt_clean else 1.0
            
            utmos_score = gt_utmos_results.get(test_fname, np.nan)
            
            sim_score = np.nan
            ref_item = ref_mapping.get(item['path'])
            if ref_item:
                ref_wav_path = os.path.join(DATA_ROOT, ref_item['path'])
                if os.path.exists(ref_wav_path) and wavlm_model is not None:
                    sim_score = calculate_sim(wavlm_bundle, wavlm_model, ref_wav_path, wav_path)

            gt_metrics.append({
                "filename": test_fname, "gt": gt_clean, "pred": pred_clean,
                "cer": cer, "wer": wer, "utmos": utmos_score, "sim": sim_score
            })

        df_gt = pd.DataFrame(gt_metrics)
        df_gt.to_csv(gt_details_path, index=False, encoding='utf-8-sig')
        
        refs_gt = df_gt['gt'].fillna("").tolist()
        hyps_gt = df_gt['pred'].fillna("").tolist()
        refs_cer = [r.replace(" ", "") for r in refs_gt]
        hyps_cer = [h.replace(" ", "") for h in hyps_gt]
        mean_cer = min(1.0, jiwer.cer(refs_cer, hyps_cer)) if refs_cer else 0.0
        mean_wer = min(1.0, jiwer.wer(refs_gt, hyps_gt)) if refs_gt else 0.0
        mean_utmos = df_gt['utmos'].mean()
        mean_sim = df_gt['sim'].mean()
        
        print(f"[GT] Result -> CER: {mean_cer:.4f}, WER: {mean_wer:.4f}, UTMOS: {mean_utmos:.4f}, SIM: {mean_sim:.4f}")
        
        final_summary.append({
            "Model": "GroundTruth", "Step": "N/A",
            "CER": mean_cer, "WER": mean_wer,
            "UTMOS": mean_utmos, "SIM": mean_sim
        })

    
    for mode in TARGET_MODES:
        dataset_name = MODE_MAP.get(mode, mode)
        
        candidates_dirs = [
            f"F5TTS_Base_vocos_KSS_1h_n2gk_{dataset_name}_lora",
            f"F5TTS_Base_vocos_custom_KSS_1h_n2gk_{dataset_name}_lora",
            f"F5TTS_Base_vocos_custom_KSS_1h_{dataset_name}_lora",
            f"F5TTS_Base_vocos_custom_KSS_1h_{mode}_lora"
        ]
        ckpt_dir = None
        for d in candidates_dirs:
            if os.path.exists(os.path.join("ckpts", d)):
                ckpt_dir = os.path.join("ckpts", d)
                break
        
        if not ckpt_dir: ckpt_dir = os.path.join("ckpts", candidates_dirs[0])
        
        if mode == "inf-h":
            tokenizer_name = "kor_efficient_allophone"
            config_path = f"src/f5_tts/configs/F5TTS_Base_ft_Lora_RTX3090_{mode}.yaml"
            if not os.path.exists(config_path): config_path = f"src/f5_tts/configs/F5TTS_Base_ft_Lora_RTX3090_{dataset_name}.yaml"
        elif mode == "inf":
            tokenizer_name = "kor_inf"
            config_path = f"src/f5_tts/configs/F5TTS_Base_ft_Lora_RTX3090_{mode}.yaml"
        elif mode == "nf":
            tokenizer_name = "kor_nf"
            config_path = f"src/f5_tts/configs/F5TTS_Base_ft_Lora_RTX3090_{mode}.yaml"
        else:
            tokenizer_name = f"kor_{MODE_MAP.get(mode, mode)}"
            config_path = f"src/f5_tts/configs/F5TTS_Base_ft_Lora_RTX3090_KSS_1h_{mode}.yaml"
            if not os.path.exists(config_path):
                config_path = f"src/f5_tts/configs/F5TTS_Base_ft_Lora_RTX3090_{mode}.yaml"

        print(f"\n==================================================")
        print(f"Evaluating Mode: {mode} (Dir: {ckpt_dir})")
        print(f"==================================================")

        # 다중 스텝 순회
        for step in EVAL_STEPS:
            model_id = f"{mode}_{step}K"
            output_dir = os.path.join(OUTPUT_BASE_DIR, model_id)
            os.makedirs(output_dir, exist_ok=True)
            details_csv_path = os.path.join(output_dir, "details.csv")
            
            existing_wavs = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
            is_generation_complete = (len(existing_wavs) >= total_items)
            
            is_result_complete = False
            if os.path.exists(details_csv_path):
                try:
                    df = pd.read_csv(details_csv_path)
                    required_cols = ['cer', 'wer', 'utmos', 'sim']
                    has_cols = all(col in df.columns for col in required_cols)
                    if len(df) == total_items and has_cols:
                        is_result_complete = True
                except: pass

            if is_result_complete:
                print(f"[{model_id}] Results complete. Skipping.")
                try:
                    df = pd.read_csv(details_csv_path)
                    refs = df['gt'].fillna("").tolist()
                    hyps = df['pred'].fillna("").tolist()
                    refs_cer = [r.replace(" ", "") for r in refs]
                    hyps_cer = [h.replace(" ", "") for h in hyps]
                    final_summary.append({
                        "Model": mode, "Step": step,
                        "CER": min(1.0, jiwer.cer(refs_cer, hyps_cer)) if refs_cer else 0.0,
                        "WER": min(1.0, jiwer.wer(refs, hyps)) if refs else 0.0,
                        "UTMOS": df['utmos'].mean(), "SIM": df['sim'].mean()
                    })
                except: pass
                continue

            if not is_generation_complete:
                print(f"[{model_id}] Generation incomplete ({len(existing_wavs)}/{total_items}). Generating missing...")
                ckpt_filename = f"model_{step}000.pt"
                ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
                
                if not os.path.exists(ckpt_path):
                    print(f"[{model_id}] Checkpoint not found: {ckpt_path}")
                    continue

                try:
                    if vocoder is None: vocoder = load_vocoder_model()
                    if not os.path.exists(config_path):
                        model_arch_config = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
                        model_cls_name = "DiT"
                    else:
                        conf = OmegaConf.load(config_path)
                        model_arch_config = conf.model.arch
                        model_cls_name = conf.model.backbone
                    model_cls = get_class(f"f5_tts.model.{model_cls_name}")
                    model = load_model(
                        model_cls=model_cls, model_cfg=model_arch_config,
                        ckpt_path=ckpt_path, mel_spec_type=VOCODER_NAME,
                        vocab_file="ckpts/pretrained/vocab_pretr.txt",
                        device=DEVICE, use_ema=True, tokenizer=tokenizer_name, use_n2gk_plus=True
                    )
                except Exception as e:
                    print(f"[{model_id}] Error loading model: {e}")
                    continue

                for item in tqdm(test_items, desc="Generating"):
                    test_fname = os.path.basename(item['path'])
                    output_wav_path = os.path.join(output_dir, test_fname)
                    if os.path.exists(output_wav_path): continue
                    
                    ref_item = ref_mapping.get(item['path'])
                    ref_audio, ref_text = None, None
                    if ref_item:
                        if ref_item['path'] in ref_cache:
                            ref_audio, ref_text = ref_cache[ref_item['path']]
                        else:
                            try:
                                ref_path_full = os.path.join(DATA_ROOT, ref_item['path'])
                                ref_audio, ref_text = preprocess_ref_audio_text(ref_path_full, ref_item['text'], show_info=lambda x: None)
                                ref_cache[ref_item['path']] = (ref_audio, ref_text)
                            except: pass
                    if ref_audio is None:
                        try:
                            ref_audio, ref_text = preprocess_ref_audio_text(os.path.join(DATA_ROOT, "1/1_0001.wav"), "Fallback ref.")
                        except: pass

                    try:
                        audio, sr, _ = infer_process(
                            ref_audio, ref_text, item['text'],
                            model, vocoder, mel_spec_type=VOCODER_NAME,
                            target_rms=TARGET_RMS, cross_fade_duration=CROSS_FADE_DURATION,
                            nfe_step=NFE_STEP, cfg_strength=CFG_STRENGTH,
                            sway_sampling_coef=SWAY_SAMPLING_COEF, speed=SPEED,
                            device=DEVICE, show_info=lambda x: None,
                            progress=None
                        )
                        sf.write(output_wav_path, audio, sr)
                    except Exception as e:
                        print(f"Failed to generate {test_fname}: {e}")
                del model
                torch.cuda.empty_cache()
            else:
                print(f"[{model_id}] Generation complete. Proceeding to evaluation.")

            if whisper_model is None: whisper_model = load_whisper_model()
            
            should_run_utmos = True
            if should_run_utmos and utmos_predictor is None: utmos_predictor = load_utmos_model()
            
            should_run_sim = True
            if should_run_sim and wavlm_model is None: wavlm_bundle, wavlm_model = load_wavlm_sim_model()

            actual_run_utmos = should_run_utmos and (utmos_predictor is not None)
            actual_run_sim = should_run_sim and (wavlm_model is not None)

            # Bulk UTMOS Prediction
            utmos_results = {}
            if actual_run_utmos:
                try:
                    print(f"[{model_id}] Running bulk UTMOS prediction on {output_dir}...")
                    bulk_preds = utmos_predictor.predict(input_dir=output_dir, device=DEVICE, verbose=False)
                    for p in bulk_preds:
                        fname = os.path.basename(p['file_path'])
                        utmos_results[fname] = p['predicted_mos']
                except Exception as e:
                    print(f"UTMOS Bulk Error: {e}")

            eval_metrics = []
            print(f"[{model_id}] Calculating Metrics (UTMOS={actual_run_utmos}, SIM={actual_run_sim})...")
            
            for item in tqdm(test_items, desc="Evaluating"):
                test_fname = os.path.basename(item['path'])
                output_wav_path = os.path.join(output_dir, test_fname)
                if not os.path.exists(output_wav_path): continue
                
                try:
                    asr_out = whisper_model.transcribe(output_wav_path, language='ko', temperature=0.0)
                    pred_text = asr_out['text']
                except:
                    print(f"Whisper Error: {e}")
                    pred_text = ""
                
                gt_clean = post_process_for_metric(normalize_n2gk_plus(item['text']))
                pred_clean = post_process_for_metric(normalize_n2gk_plus(pred_text))
                gt_ns = gt_clean.replace(" ", "")
                pred_ns = pred_clean.replace(" ", "")
                
                cer = min(1.0, jiwer.cer(gt_ns, pred_ns)) if gt_ns else 1.0
                wer = min(1.0, jiwer.wer(gt_clean, pred_clean)) if gt_clean else 1.0
                
                # Get UTMOS from bulk results
                utmos_score = utmos_results.get(test_fname, np.nan)
                
                sim_score = np.nan
                if actual_run_sim:
                    ref_item = ref_mapping.get(item['path'])
                    if ref_item:
                        ref_wav_path = os.path.join(DATA_ROOT, ref_item['path'])
                        if os.path.exists(ref_wav_path):
                            sim_score = calculate_sim(wavlm_bundle, wavlm_model, ref_wav_path, output_wav_path)
                
                eval_metrics.append({
                    "filename": test_fname, "gt": gt_clean, "pred": pred_clean,
                    "cer": cer, "wer": wer, "utmos": utmos_score, "sim": sim_score
                })

            df_res = pd.DataFrame(eval_metrics)
            df_res.to_csv(details_csv_path, index=False, encoding='utf-8-sig')
            
            refs = df_res['gt'].fillna("").tolist()
            hyps = df_res['pred'].fillna("").tolist()
            refs_cer = [r.replace(" ", "") for r in refs]
            hyps_cer = [h.replace(" ", "") for h in hyps]
            mean_cer = min(1.0, jiwer.cer(refs_cer, hyps_cer)) if refs_cer else 0.0
            mean_wer = min(1.0, jiwer.wer(refs, hyps)) if refs else 0.0
            mean_utmos = df_res['utmos'].mean()
            mean_sim = df_res['sim'].mean()
            
            print(f"[{model_id}] Result -> CER: {mean_cer:.4f}, WER: {mean_wer:.4f}, UTMOS: {mean_utmos:.4f}, SIM: {mean_sim:.4f}")
            
            final_summary.append({
                "Model": mode, "Step": step,
                "CER": mean_cer, "WER": mean_wer,
                "UTMOS": mean_utmos if actual_run_utmos else "",
                "SIM": mean_sim if actual_run_sim else ""
            })

    if final_summary:
        df_summary = pd.DataFrame(final_summary)
        print("\n" + "="*80)
        print("FINAL VALIDATION REPORT")
        print("="*80)
        print(df_summary.to_string(index=False))
        save_path = os.path.join(OUTPUT_BASE_DIR, "validation_summary_comprehensive.csv")
        df_summary.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\nReport saved to: {save_path}")

if __name__ == "__main__":
    run_evaluation()
