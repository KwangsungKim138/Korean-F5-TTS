import os
import sys
import torch
import torchaudio
import soundfile as sf
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
import whisper
import jiwer
import re
from omegaconf import OmegaConf
from hydra.utils import get_class

# Add src to path to import f5_tts modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

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

# Define the models to evaluate
# type: "inference" (default) | "existing" | "ground_truth"
# - inference: ckpt_path, model_cfg, vocab_file, tokenizer 필요
# - existing: path (오디오 파일들이 있는 폴더 경로) 필요
# - ground_truth: path (데이터셋 루트 경로, 예: data/KSS) 필요
MODELS = [
    # 1. Ground Truth
    {
        "name": "Ground_Truth",
        "type": "ground_truth",
        "path": "data/KSS" # 원본 오디오 루트 (data/KSS/1/1_0001.wav 등을 찾음)
    },
    # 2. Inference Models
    {
        "type": "existing",
        "path": "eval_results/F5TTS_Lora_A100_grapheme_55K",
        "name": "F5TTS_Lora_A100_grapheme_55K",
        "ckpt_path": "tests/model_55K_grp.pt",
        "vocab_file": "ckpts/pretrained/vocab_pretr.txt",
        "model_cfg": "src/f5_tts/configs/F5TTS_Base_ft_Lora_A100_grapheme.yaml",
        "tokenizer": "kor_grapheme"
    },
    {
        "type": "existing",
        "path": "eval_results/F5TTS_Lora_A100_grapheme_60K",
        "name": "F5TTS_Lora_A100_grapheme_60K",
        "ckpt_path": "tests/model_60K_grp.pt",
        "vocab_file": "ckpts/pretrained/vocab_pretr.txt",
        "model_cfg": "src/f5_tts/configs/F5TTS_Base_ft_Lora_A100_grapheme.yaml",
        "tokenizer": "kor_grapheme"
    },
    {
        "type": "existing",
        "path": "eval_results/F5TTS_Lora_A100_grapheme_57500",
        "name": "F5TTS_Lora_A100_grapheme_57500",
        "ckpt_path": "tests/model_57500.pt",
        "vocab_file": "ckpts/pretrained/vocab_pretr.txt",
        "model_cfg": "src/f5_tts/configs/F5TTS_Base_ft_Lora_A100_grapheme.yaml",
        "tokenizer": "kor_grapheme"
    },
    # 3. Existing Folder Example (이미 생성된 오디오가 있다면 주석 해제)
    # {
    #     "name": "My_Previous_Result",
    #     "type": "existing",
    #     "path": "eval_results/F5TTS_Old_Model" 
    # }
]

# Settings
TEST_SET_PATH = "data/KSS/test.txt"
# If not exists, check fallback... (생략, 기존 로직 유지)
if not os.path.exists(TEST_SET_PATH):
    print(f"Warning: {TEST_SET_PATH} not found.")
    throw

OUTPUT_BASE_DIR = "eval_results"

# Common settings (matching batch_infer.py)
REF_AUDIO = "data/KSS/1/1_0001.wav"
REF_TEXT = "그녀의 사랑을 얻기 위해 애썼지만, 헛수고였다."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCODER_NAME = "vocos"
TARGET_RMS = 0.1
CROSS_FADE_DURATION = 0.15
NFE_STEP = 32
CFG_STRENGTH = 2.0
SWAY_SAMPLING_COEF = -1.0
SPEED = 1.0

# Whisper Settings
WHISPER_SIZE = "large-v3"

# --------------------------
# Helper Functions
# --------------------------

def post_process_for_metric(text):
    """get_cer_wer.py의 전처리 로직과 동일"""
    # 특수문자 제거 (한글, 공백, 로마 알파벳, 숫자만)
    text = re.sub(r'[^\w\s가-힣ㄱ-ㅎ]', '', text)
    # 다중 공백을 단일 공백으로 치환
    text = re.sub(r'\s+', ' ', text)
    # Whisper 단골 hallucination
    text = re.sub(r'((자막 제공 및\s)+광고를 포함하고 있습니다)|(이 영상은 한국국토정보공사의 자료입니다)', '', text).strip()
    return text

def load_test_sentences(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("|")
            
            if len(parts) >= 3:
                filename_raw = parts[0]
                text = parts[2] # Normalized text
            elif len(parts) == 2:
                filename_raw = parts[0]
                text = parts[1] # Raw text
            else:
                filename_raw = f"sample_{len(items):04d}.wav"
                text = line

            if "/" in filename_raw:
                filename = filename_raw.split("/")[-1]
            else:
                filename = filename_raw
            
            if not filename.endswith(".wav"):
                filename += ".wav"
                
            items.append({
                "filename_raw": filename_raw,
                "filename": filename,
                "text": text
            })
    return items

def run_evaluation():
    # 1. Load UTMOS
    print("Loading UTMOS predictor...")
    try:
        utmos_predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        utmos_predictor = utmos_predictor.to(DEVICE)
    except Exception as e:
        print(f"Error loading UTMOS: {e}")
        return

    # 2. Load Whisper
    print(f"Loading Whisper model ({WHISPER_SIZE})...")
    try:
        whisper_model = whisper.load_model(WHISPER_SIZE, device=DEVICE)
    except Exception as e:
        print(f"Error loading Whisper: {e}")
        return

    # Load Test Items
    test_items = load_test_sentences(TEST_SET_PATH)
    print(f"Found {len(test_items)} test sentences.")

    # Shared Resources
    vocoder = None
    ref_audio_processed = None
    ref_text_processed = None
    
    final_summary_rows = []

    for model_config in MODELS:
        model_name = model_config['name']
        model_type = model_config.get('type', 'inference')
        print(f"\n[{model_name}] Processing (Type: {model_type})...")

        output_dir = os.path.join(OUTPUT_BASE_DIR, model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # ---------------------------------------------------------
        # A. Audio Generation / Preparation
        # ---------------------------------------------------------
        files_to_eval = [] # List of (path, gt_text)

        # ==========================================
        # CASE 1: Ground Truth
        # ==========================================
        #  files_to_eval에 (audio_path, gt_text) 튜플을 저장
        
        if model_type == "ground_truth":
            dataset_root = model_config.get('path', 'data/KSS')
            print(f"  Locating Ground Truth files in {dataset_root}...")
            
            found_count = 0
            for item in test_items:
                gt_path = os.path.join(dataset_root, item['filename_raw'])
                if not os.path.exists(gt_path):
                    gt_path = os.path.join(dataset_root, item['filename'])
                if os.path.exists(gt_path):
                    files_to_eval.append((gt_path, item['text']))
                    found_count += 1
            print(f"  Found {found_count}/{len(test_items)} Ground Truth files.")

        # ==========================================
        # CASE 2: Existing Folder
        # ==========================================
        elif model_type == "existing":
            existing_path = model_config.get('path', '')
            print(f"  Using existing audio files from {existing_path}...")
            
            if not os.path.exists(existing_path):
                print(f"  Error: Directory {existing_path} does not exist.")
                continue
                
            found_count = 0
            for item in test_items:
                # Look for filename (basename) in the folder
                target_path = os.path.join(existing_path, item['filename'])
                if os.path.exists(target_path):
                    files_to_eval.append((target_path, item['text']))
                    found_count += 1
            
            print(f"  Found {found_count}/{len(test_items)} existing files.")

        # ==========================================
        # CASE 3: Inference
        # ==========================================
        else: # type == "inference"
            # Initialize shared resources if needed
            if vocoder is None:
                print(f"  Loading Vocoder ({VOCODER_NAME})...")
                vocoder = load_vocoder(vocoder_name=VOCODER_NAME, is_local=False, device=DEVICE)
            
            if ref_audio_processed is None:
                print("  Processing Reference Audio...")
                if not os.path.exists(REF_AUDIO):
                    print(f"  Error: Reference audio {REF_AUDIO} not found.")
                    continue
                try:
                    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(REF_AUDIO, REF_TEXT)
                except Exception as e:
                    print(f"  Error processing reference audio: {e}")
                    continue

            # Checkpoint check
            ckpt_path = model_config['ckpt_path']
            if not os.path.exists(ckpt_path):
                print(f"  [Skipping] Checkpoint not found: {ckpt_path}")
                parent_dir = os.path.dirname(ckpt_path)
                if os.path.exists(parent_dir):
                    pts = list(Path(parent_dir).rglob("*.pt"))
                    if pts:
                        print(f"  Found alternative checkpoint: {pts[0]}")
                        ckpt_path = str(pts[0])
                    else:
                        continue
                else:
                    continue

            # Load Model
            try:
                cfg_path = model_config.get("model_cfg", "")
                if cfg_path and os.path.exists(cfg_path):
                    conf = OmegaConf.load(cfg_path)
                    model_arch_config = conf.model.arch
                    model_cls_name = conf.model.backbone
                else:
                    print(f"  Warning: Config file {cfg_path} not found. Using default parameters.")
                    model_arch_config = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
                    model_cls_name = "DiT"
                
                model_cls = get_class(f"f5_tts.model.{model_cls_name}")
                
                vocab_file = model_config.get('vocab_file', "")
                tokenizer_type = model_config.get('tokenizer', "custom")
                
                model = load_model(
                    model_cls=model_cls,
                    model_cfg=model_arch_config,
                    ckpt_path=ckpt_path,
                    mel_spec_type=VOCODER_NAME,
                    vocab_file=vocab_file,
                    device=DEVICE,
                    use_ema=True,
                    tokenizer=tokenizer_type,
                    use_n2gk_plus=True
                )
            except Exception as e:
                print(f"  Error loading model {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Generate Audio
            print(f"  Generating {len(test_items)} sentences...")
            
            for i, item in enumerate(tqdm(test_items)):
                output_filename = os.path.basename(item["filename"])
                gen_text = item["text"]
                output_file = os.path.join(output_dir, output_filename)
                
                # Add to evaluation list (whether newly generated or skipped)
                
                if os.path.exists(output_file):
                     files_to_eval.append((output_file, gen_text))
                     continue

                try:
                    audio, sr, spectrogram = infer_process(
                        ref_audio_processed,
                        ref_text_processed,
                        gen_text,
                        model,
                        vocoder,
                        mel_spec_type=VOCODER_NAME,
                        target_rms=TARGET_RMS,
                        cross_fade_duration=CROSS_FADE_DURATION,
                        nfe_step=NFE_STEP,
                        cfg_strength=CFG_STRENGTH,
                        sway_sampling_coef=SWAY_SAMPLING_COEF,
                        speed=SPEED,
                        device=DEVICE,
                    )
                    sf.write(output_file, audio, sr)
                    files_to_eval.append((output_file, gen_text))
                except Exception as e:
                    print(f"    Error generating {output_filename}: {e}")
            
            # Filter files that actually exist (in case generation failed)
            files_to_eval = [f for f in files_to_eval if os.path.exists(f[0])]

        # ---------------------------------------------------------
        # B. Evaluation Loop (UTMOS + CER + WER)
        # ---------------------------------------------------------
        if not files_to_eval:
            print("No files to evaluate.")
            continue

        print(f"  Evaluating metrics for {len(files_to_eval)} files...")
        
        results = []
        
        gt_texts_no_space = [] # For Global CER
        pred_texts_no_space = []
        gt_texts_list = [] # For Global WER
        pred_texts_list = []

        for audio_path, raw_gt_text in tqdm(files_to_eval, desc="Evaluating"):
            # 1. UTMOS
            try:
                import librosa
                wav, sr = librosa.load(audio_path, sr=None, mono=True)
                wav_tensor = torch.from_numpy(wav).to(DEVICE).unsqueeze(0)
                with torch.no_grad():
                    utmos_score = utmos_predictor(wav_tensor, sr).item()
            except:
                utmos_score = 0.0

            # 2. Whisper ASR
            try:
                asr_res = whisper_model.transcribe(audio_path, language='ko', temperature=0.0)
                raw_pred_text = asr_res['text']
            except Exception as e:
                print(f"ASR Error {audio_path}: {e}")
                raw_pred_text = ""

            # 3. Post-process (N2gk+ -> Clean)
            # GT
            gt_n2gk = normalize_n2gk_plus(raw_gt_text)
            gt_final = post_process_for_metric(gt_n2gk)
            gt_final_ns = gt_final.replace(' ', '') # No space
            
            # Pred
            pred_n2gk = normalize_n2gk_plus(raw_pred_text)
            pred_final = post_process_for_metric(pred_n2gk)
            pred_final_ns = pred_final.replace(' ', '') # No space

            # 4. Calculate Individual Metrics
            cer = min(1.0, jiwer.cer(gt_final_ns, pred_final_ns)) if gt_final_ns else 1.0
            wer = min(1.0, jiwer.wer(gt_final, pred_final)) if gt_final else 1.0
            
            results.append({
                "filename": os.path.basename(audio_path),
                "utmos": utmos_score,
                "cer": cer,
                "wer": wer,
                "gt_ns": gt_final_ns, # Saved for micro-avg calculation
                "pred_ns": pred_final_ns,
                "gt_clean": gt_final,
                "pred_clean": pred_final
            })

        # ---------------------------------------------------------
        # C. Statistics & Summary
        # ---------------------------------------------------------
        df = pd.DataFrame(results)
        
        # 1. Global Metrics (Micro-average using jiwer on full list)
        global_cer = min(1.0, jiwer.cer(df['gt_ns'].tolist(), df['pred_ns'].tolist()))
        global_wer = min(1.0, jiwer.wer(df['gt_clean'].tolist(), df['pred_clean'].tolist()))
        
        mean_utmos = df['utmos'].mean()

        # 2. SFR (Synthesis Failure Rate): CER > 0.5
        fail_count = len(df[df['cer'] > 0.5])
        total_count = len(df)
        sfr = (fail_count / total_count) * 100 if total_count > 0 else 0.0

        # 3. Valid CER (Micro-average using jiwer on valid list)
        valid_df = df[df['cer'] <= 0.5]
        if not valid_df.empty:
            valid_mean_cer = min(1.0, jiwer.cer(valid_df['gt_ns'].tolist(), valid_df['pred_ns'].tolist()))
        else:
            valid_mean_cer = 0.0
            
        # 4. Worst Case
        worst_row = df.loc[df['cer'].idxmax()] if not df.empty else None

        # Print Model Summary
        print(f"\n[Result: {model_name}]")
        print(f"  UTMOS: {mean_utmos:.4f}")
        print(f"  CER(Global): {global_cer:.4f}")
        print(f"  WER: {global_wer:.4f}")
        print(f"  CER(Valid): {valid_mean_cer:.4f}")
        print(f"  SFR: {sfr:.2f}% ({fail_count}/{total_count} failed)")
        
        if worst_row is not None:
            print(f"  Worst Case (CER: {worst_row['cer']:.4f}):")
            print(f"    GT  : {worst_row['gt_clean']}")
            print(f"    Pred: {worst_row['pred_clean']}")

        # Save Details
        df.to_csv(os.path.join(output_dir, "details.csv"), index=False, encoding="utf-8-sig")
        
        final_summary_rows.append({
            "Model": model_name,
            "CER(Global)": f"{global_cer:.4f}",
            "WER": f"{global_wer:.4f}",
            "CER(Valid)": f"{valid_mean_cer:.4f}",
            "SFR (%)": f"{sfr:.2f}",
            "UTMOS": f"{mean_utmos:.4f}",
        })

    # ---------------------------------------------------------
    # D. Final Report Table
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("FINAL EVALUATION SUMMARY")
    print("="*80)
    summary_df = pd.DataFrame(final_summary_rows)
    print(summary_df.to_string(index=False))
    
    summary_path = os.path.join(OUTPUT_BASE_DIR, "final_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    run_evaluation()