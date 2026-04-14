"""
CoreaSpeech / KUB-style test list evaluation (Whisper CER/WER, UTMOS, WavLM SIM).

Evaluates multiple TTS models and checkpoints automatically (Grid Search) or manually.

Test list format (pipe-separated, 6 columns, one utterance per line):
  subset | ref_wav | ref_text | ref_duration | gt_wav | target_text

- subset: Category of the test item (e.g., clean, noisy, numeric)
- ref_wav: Path to the reference audio (relative to --data-root) for voice cloning
- ref_text: Transcript of the reference audio
- ref_duration: Duration of the reference audio in seconds (used as a duration hint)
- gt_wav: Path to the ground truth audio (relative to --data-root) for SIM comparison
- target_text: The text to be synthesized by the TTS model

Note: The script performs a direct 1:1 evaluation for each line.
"""

import os
import sys
import warnings
import re
import argparse
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
import torchaudio

warnings.filterwarnings("ignore", message="Plan failed with a cudnnException")
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

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

try:
    from f5_tts.train.datasets.normalization_n2gk import normalize_n2gk_plus
    print("✅ N2gkPlus normalization module loaded successfully.")
except ImportError:
    print("❌ Warning: 'normalization_n2gk.py' not found. Normalization will be skipped.")

    def normalize_n2gk_plus(text):
        return text


# --------------------------
# Defaults (override via CLI)
# --------------------------

BEST_MODELS = {
    "grapheme": 450,
    "phoneme": 450,
    "salt_n": 450,
    "salt_vcp": 450,
}

MODE_MAP = {
    "V+N+L-H": "efficient_allophone",
    "V+C+J": "allophone",
    "N": "n_only",
    "V": "i_only",
    "C": "c_only",
    "V+C": "i_and_c",
    "V+N": "i_and_n",
    "N+L": "nf",
    "V+N+L": "inf",
    "salt_n": "n_only",
    "salt_vcp": "allophone",
}

# Mode -> Hydra dataset name suffix (ckpts/...CoreaSpeech_<name>_lora)
MODE_DATASET = {
    "grapheme": "CoreaSpeech_grapheme",
    "phoneme": "CoreaSpeech_phoneme",
    "salt_n": "CoreaSpeech_salt_n",
    "salt_vcp": "CoreaSpeech_salt_vcp",
}

DEFAULT_DATA_ROOT = os.path.join("data", "CoreaSpeech_kub")
DEFAULT_TEST_TXT = os.path.join(DEFAULT_DATA_ROOT, "test.txt")
DEFAULT_OUTPUT_DIR = "CoreaSpeech_KUB_eval"
DEFAULT_VOCAB = "ckpts/pretrained/vocab_pretr.txt"

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
# KUB list parsing & pair reference mapping
# --------------------------


def parse_KUB_line(line: str):
    """
    New KUB format: subset | ref_wav | ref_text | ref_duration | gt_wav | target_text
    """
    parts = line.strip().split("|")
    if len(parts) < 6:
        return None
    try:
        item = {
            "subset": parts[0].strip(),
            "ref_wav": parts[1].strip(),
            "ref_text": parts[2].strip(),
            "ref_duration": float(parts[3].strip()),
            "gt_wav": parts[4].strip(),
            "target_text": parts[5].strip(),
        }
        return item
    except ValueError:
        return None


def load_test_items(test_path: str):
    test_items = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            item = parse_KUB_line(line)
            if item:
                test_items.append(item)
    return test_items


# --------------------------
# Metrics helpers
# --------------------------


def post_process_for_metric(text):
    text = re.sub(r"[^\w\s가-힣ㄱ-ㅎ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_vocoder_model():
    print(f"Loading Vocoder ({VOCODER_NAME})...")
    return load_vocoder(vocoder_name=VOCODER_NAME, is_local=False, device=DEVICE)


def load_whisper_model():
    print(f"Loading Whisper ({WHISPER_SIZE})...")
    return whisper.load_model(WHISPER_SIZE, device=DEVICE)


def load_utmos_model():
    print("Loading UTMOSv2 predictor...")
    if utmosv2 is None:
        return None
    try:
        model = utmosv2.create_model(pretrained=True)
        return model.eval().to(DEVICE)
    except Exception as e:
        print(f"Failed to load UTMOSv2: {e}")
        return None


def load_wavlm_sim_model():
    print("Loading WavLM (torchaudio bundle)...")
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
            
            # cosine_similarity returns a tensor of shape [batch_size] (here [1] or [2] if stereo)
            # We take the mean to get a single scalar value
            sim = F.cosine_similarity(ref_emb, gen_emb, dim=-1).mean().item()
        return sim
    except Exception as e:
        print(f"SIM Error: {e}")
        return np.nan


def resolve_audio_path(data_root: str, rel_path: str) -> str:
    return os.path.join(data_root, rel_path)


def find_ckpt_dir(mode: str, dataset_name: str) -> str | None:
    candidates_dirs = [
        f"F5TTS_Base_vocos_custom_{dataset_name}_lora",
        f"F5TTS_Base_{VOCODER_NAME}_custom_{dataset_name}_lora",
        f"F5TTS_Base_vocos_{dataset_name}_lora",
    ]
    for d in candidates_dirs:
        p = os.path.join("ckpts", d)
        if os.path.isdir(p):
            return p
    return None


def mode_to_tokenizer(mode: str) -> str:
    if mode == "grapheme":
        return "kor_grapheme"
    if mode == "phoneme":
        return "kor_phoneme"
    if mode == "inf":
        return "kor_inf"
    if mode == "nf":
        return "kor_nf"
    if mode == "inf-h":
        return "kor_efficient_allophone"
    return f"kor_{MODE_MAP.get(mode, mode)}"


def default_config_path(mode: str, dataset_name: str) -> str | None:
    # Prefer RTX3090 CoreaSpeech configs when present
    base = f"src/f5_tts/configs/F5TTS_Base_ft_Lora_RTX3090_{dataset_name}.yaml"
    if os.path.exists(base):
        return base
    alt = f"src/f5_tts/configs/F5TTS_Base_ft_Lora_A100_{mode}.yaml"
    if os.path.exists(alt):
        return alt
    return None


def extract_step(ckpt_path):
    basename = os.path.basename(ckpt_path)
    match = re.search(r"model_(\d+)\.(pt|safetensors)", basename)
    if match:
        return int(match.group(1))
    return 0

def run_evaluation(args):
    data_root = os.path.abspath(args.data_root)
    test_path = args.test_txt
    output_base = args.output_dir
    vocab_file = args.vocab

    if not os.path.isfile(test_path):
        print(f"Error: test list not found: {test_path}")
        return

    test_items = load_test_items(test_path)
    total_items = len(test_items)
    if total_items == 0:
        print("Error: no valid lines in test list.")
        return

    # Build evaluation tasks: list of (mode, step, ckpt_path)
    eval_tasks = []
    
    if args.ckpt_paths or args.ckpt_dir:
        # Manual override mode
        mode = args.modes[0]
        ckpts = []
        if args.ckpt_paths:
            ckpts = args.ckpt_paths
        else:
            for root, _, files in os.walk(args.ckpt_dir):
                for f in files:
                    if f.endswith(".pt") or f.endswith(".safetensors"):
                        ckpts.append(os.path.join(root, f))
        ckpts = sorted(ckpts, key=extract_step)
        for c in ckpts:
            eval_tasks.append((mode, extract_step(c), c))
    else:
        # Automatic Grid Search mode
        for mode in args.modes:
            dataset_name = MODE_DATASET.get(mode, f"CoreaSpeech_{MODE_MAP.get(mode, mode)}")
            ckpt_dir = find_ckpt_dir(mode, dataset_name)
            if not ckpt_dir:
                print(f"[{mode}] Checkpoint dir not found. Expected something like ckpts/*{dataset_name}*_lora")
                continue
            
            for step_k in args.steps:
                target_pt = f"model_{step_k}000.pt"
                target_st = f"model_{step_k}000.safetensors"
                found_path = None
                
                for root, _, files in os.walk(ckpt_dir):
                    if target_pt in files:
                        found_path = os.path.join(root, target_pt)
                        break
                    elif target_st in files:
                        found_path = os.path.join(root, target_st)
                        break
                
                if found_path:
                    eval_tasks.append((mode, step_k * 1000, found_path))
                else:
                    print(f"[{mode}] Checkpoint for {step_k}K not found in {ckpt_dir}")

    if not eval_tasks:
        print("No checkpoints found to evaluate.")
        return

    vocoder = None
    whisper_model = None
    utmos_predictor = None
    wavlm_bundle, wavlm_model = None, None

    ref_cache = {}
    all_results = []

    # --------------------------
    # Ground Truth pass
    # --------------------------
    gt_dir = os.path.join(output_base, "GT")
    gt_wav_dir = os.path.join(gt_dir, "wavs")
    gt_details_path = os.path.join(gt_dir, "details.csv")
    os.makedirs(gt_wav_dir, exist_ok=True)

    gt_done = False
    if os.path.exists(gt_details_path):
        try:
            df = pd.read_csv(gt_details_path)
            if len(df) == total_items:
                gt_done = True
                print("[GT] Results found. Loading...")
                for _, row in df.iterrows():
                    all_results.append({
                        "mode": "GT",
                        "step": "GT",
                        "subset": row["subset"],
                        "cer": row["cer"],
                        "wer": row["wer"],
                        "utmos": row["utmos"],
                        "sim": row["sim"],
                    })
        except Exception:
            pass

    if not gt_done:
        print("\n" + "=" * 50)
        print("Evaluating Ground Truth (GT)")
        print("=" * 50)
        print(f"[GT] Preparing audio symlinks in {gt_wav_dir}...")
        for item in test_items:
            src_path = resolve_audio_path(data_root, item["gt_wav"])
            dst_path = os.path.join(gt_wav_dir, os.path.basename(item["gt_wav"]))
            if not os.path.exists(dst_path):
                try:
                    os.symlink(os.path.abspath(src_path), dst_path)
                except OSError:
                    import shutil
                    shutil.copy(src_path, dst_path)

        if whisper_model is None:
            whisper_model = load_whisper_model()
        if utmos_predictor is None:
            utmos_predictor = load_utmos_model()
        if wavlm_model is None:
            wavlm_bundle, wavlm_model = load_wavlm_sim_model()

        gt_utmos_results = {}
        if utmos_predictor is not None:
            print("[GT] Running bulk UTMOS prediction...")
            try:
                bulk_preds = utmos_predictor.predict(input_dir=gt_wav_dir, device=DEVICE, verbose=False)
                for p in bulk_preds:
                    fname = os.path.basename(p["file_path"])
                    gt_utmos_results[fname] = p["predicted_mos"]
            except Exception as e:
                print(f"GT UTMOS Error: {e}")

        gt_metrics = []
        for item in tqdm(test_items, desc="Evaluating GT"):
            test_fname = os.path.basename(item["gt_wav"])
            wav_path = os.path.join(gt_wav_dir, test_fname)
            try:
                asr_out = whisper_model.transcribe(wav_path, language="ko", temperature=0.0)
                pred_text = asr_out["text"]
            except Exception as e:
                print(f"Whisper Error: {e}")
                pred_text = ""

            gt_clean = post_process_for_metric(normalize_n2gk_plus(item["target_text"]))
            pred_clean = post_process_for_metric(normalize_n2gk_plus(pred_text))
            gt_ns = gt_clean.replace(" ", "")
            pred_ns = pred_clean.replace(" ", "")
            cer = min(1.0, jiwer.cer(gt_ns, pred_ns)) if gt_ns else 1.0
            wer = min(1.0, jiwer.wer(gt_clean, pred_clean)) if gt_clean else 1.0
            utmos_score = gt_utmos_results.get(test_fname, np.nan)
            sim_score = np.nan
            
            ref_wav_path = resolve_audio_path(data_root, item["ref_wav"])
            if os.path.exists(ref_wav_path) and wavlm_model is not None:
                sim_score = calculate_sim(wavlm_bundle, wavlm_model, ref_wav_path, wav_path)
            
            gt_metrics.append({
                "filename": test_fname,
                "subset": item["subset"],
                "gt": gt_clean,
                "pred": pred_clean,
                "cer": cer,
                "wer": wer,
                "utmos": utmos_score,
                "sim": sim_score,
            })
            all_results.append({
                "mode": "GT",
                "step": "GT",
                "subset": item["subset"],
                "cer": cer,
                "wer": wer,
                "utmos": utmos_score,
                "sim": sim_score,
            })

        df_gt = pd.DataFrame(gt_metrics)
        df_gt.to_csv(gt_details_path, index=False, encoding="utf-8-sig")

    # --------------------------
    # Model evaluation
    # --------------------------
    for mode, step, ckpt_path in eval_tasks:
        tokenizer_name = mode_to_tokenizer(mode)
        dataset_name = MODE_DATASET.get(mode, f"CoreaSpeech_{MODE_MAP.get(mode, mode)}")
        config_path = default_config_path(mode, dataset_name)
        
        model_id = f"{mode}_{step // 1000}K"
        output_dir = os.path.join(output_base, model_id)
        os.makedirs(output_dir, exist_ok=True)
        details_csv_path = os.path.join(output_dir, "details.csv")

        existing_wavs = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
        is_generation_complete = len(existing_wavs) >= total_items

        is_result_complete = False
        if os.path.exists(details_csv_path):
            try:
                df = pd.read_csv(details_csv_path)
                required_cols = ["cer", "wer", "utmos", "sim"]
                if len(df) == total_items and all(c in df.columns for c in required_cols):
                    is_result_complete = True
                    print(f"[{model_id}] Results complete. Skipping generation and eval.")
                    for _, row in df.iterrows():
                        all_results.append({
                            "mode": mode,
                            "step": step,
                            "subset": row["subset"],
                            "cer": row["cer"],
                            "wer": row["wer"],
                            "utmos": row["utmos"],
                            "sim": row["sim"],
                        })
            except Exception:
                pass

        if is_result_complete:
            continue

        if not is_generation_complete:
            print(f"\n[{model_id}] Generating ({len(existing_wavs)}/{total_items})...")
            try:
                if vocoder is None:
                    vocoder = load_vocoder_model()
                if config_path and os.path.isfile(config_path):
                    conf = OmegaConf.load(config_path)
                    model_arch_config = conf.model.arch
                    model_cls_name = conf.model.backbone
                else:
                    model_arch_config = dict(
                        dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,
                    )
                    model_cls_name = "DiT"
                model_cls = get_class(f"f5_tts.model.{model_cls_name}")
                model = load_model(
                    model_cls=model_cls,
                    model_cfg=model_arch_config,
                    ckpt_path=ckpt_path,
                    mel_spec_type=VOCODER_NAME,
                    vocab_file=vocab_file,
                    device=DEVICE,
                    use_ema=True,
                    tokenizer=tokenizer_name,
                    use_n2gk_plus=True,
                )
            except Exception as e:
                print(f"[{model_id}] Error loading model: {e}")
                continue

            for item in tqdm(test_items, desc="Generating"):
                test_fname = os.path.basename(item["gt_wav"])
                output_wav_path = os.path.join(output_dir, test_fname)
                if os.path.exists(output_wav_path):
                    continue

                ref_audio, ref_text = None, None
                if item["ref_wav"] in ref_cache:
                    ref_audio, ref_text = ref_cache[item["ref_wav"]]
                else:
                    try:
                        ref_path_full = resolve_audio_path(data_root, item["ref_wav"])
                        ref_audio, ref_text = preprocess_ref_audio_text(
                            ref_path_full, item["ref_text"], show_info=lambda x: None
                        )
                        ref_cache[item["ref_wav"]] = (ref_audio, ref_text)
                    except Exception:
                        pass

                if ref_audio is None:
                    continue

                try:
                    audio, sr, _ = infer_process(
                        ref_audio,
                        ref_text,
                        item["target_text"],
                        model,
                        vocoder,
                        mel_spec_type=VOCODER_NAME,
                        target_rms=TARGET_RMS,
                        cross_fade_duration=CROSS_FADE_DURATION,
                        nfe_step=NFE_STEP,
                        cfg_strength=CFG_STRENGTH,
                        sway_sampling_coef=SWAY_SAMPLING_COEF,
                        speed=SPEED,
                        fix_duration=None,
                        device=DEVICE,
                        show_info=lambda x: None,
                        progress=None,
                    )
                    sf.write(output_wav_path, audio, sr)
                except Exception as e:
                    print(f"Failed to generate {test_fname}: {e}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"\n[{model_id}] Generation complete. Evaluating.")

        if whisper_model is None:
            whisper_model = load_whisper_model()
        if utmos_predictor is None:
            utmos_predictor = load_utmos_model()
        if wavlm_model is None:
            wavlm_bundle, wavlm_model = load_wavlm_sim_model()

        actual_run_utmos = utmos_predictor is not None
        actual_run_sim = wavlm_model is not None

        utmos_results = {}
        if actual_run_utmos:
            try:
                print(f"[{model_id}] Running bulk UTMOS on {output_dir}...")
                bulk_preds = utmos_predictor.predict(input_dir=output_dir, device=DEVICE, verbose=False)
                for p in bulk_preds:
                    fname = os.path.basename(p["file_path"])
                    utmos_results[fname] = p["predicted_mos"]
            except Exception as e:
                print(f"UTMOS Bulk Error: {e}")

        eval_metrics = []
        print(f"[{model_id}] Metrics (UTMOS={actual_run_utmos}, SIM={actual_run_sim})...")
        for item in tqdm(test_items, desc="Evaluating"):
            test_fname = os.path.basename(item["gt_wav"])
            output_wav_path = os.path.join(output_dir, test_fname)
            if not os.path.exists(output_wav_path):
                continue
            try:
                asr_out = whisper_model.transcribe(output_wav_path, language="ko", temperature=0.0)
                pred_text = asr_out["text"]
            except Exception as e:
                print(f"Whisper Error: {e}")
                pred_text = ""

            gt_clean = post_process_for_metric(normalize_n2gk_plus(item["target_text"]))
            pred_clean = post_process_for_metric(normalize_n2gk_plus(pred_text))
            gt_ns = gt_clean.replace(" ", "")
            pred_ns = pred_clean.replace(" ", "")
            cer = min(1.0, jiwer.cer(gt_ns, pred_ns)) if gt_ns else 1.0
            wer = min(1.0, jiwer.wer(gt_clean, pred_clean)) if gt_clean else 1.0
            utmos_score = utmos_results.get(test_fname, np.nan)
            sim_score = np.nan
            if actual_run_sim:
                ref_wav_path = resolve_audio_path(data_root, item["ref_wav"])
                if os.path.exists(ref_wav_path):
                    sim_score = calculate_sim(wavlm_bundle, wavlm_model, ref_wav_path, output_wav_path)
            
            eval_metrics.append({
                "filename": test_fname,
                "subset": item["subset"],
                "gt": gt_clean,
                "pred": pred_clean,
                "cer": cer,
                "wer": wer,
                "utmos": utmos_score,
                "sim": sim_score,
            })
            all_results.append({
                "mode": mode,
                "step": step,
                "subset": item["subset"],
                "cer": cer,
                "wer": wer,
                "utmos": utmos_score,
                "sim": sim_score,
            })

        df_res = pd.DataFrame(eval_metrics)
        df_res.to_csv(details_csv_path, index=False, encoding="utf-8-sig")

    if all_results:
        df_all = pd.DataFrame(all_results)
        summary = df_all.groupby(['mode', 'step', 'subset'])[['cer', 'wer', 'utmos', 'sim']].mean().reset_index()
        print("\n" + "=" * 80)
        print("FINAL EVALUATION REPORT (CoreaSpeech KUB)")
        print("=" * 80)
        print(summary.to_string(index=False))
        save_path = os.path.join(output_base, "evaluation_summary_comprehensive.csv")
        summary.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\nReport saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate F5-TTS on CoreaSpeech / KUB test list")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT, help="Root directory")
    parser.add_argument("--test-txt", type=str, default=DEFAULT_TEST_TXT, help="Test list")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--vocab", type=str, default=DEFAULT_VOCAB, help="vocab.txt for load_model")
    
    # Grid Search Arguments
    parser.add_argument("--modes", type=str, nargs="+", default=["grapheme", "phoneme", "salt_n", "salt_vcp"], help="Modes to evaluate")
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 150, 200, 250, 300, 350, 400, 450], help="Steps to evaluate (in thousands, e.g., 100 for 100000)")
    
    # Manual Override Arguments
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Manual override: Directory containing checkpoints")
    parser.add_argument("--ckpt_paths", type=str, nargs="+", default=None, help="Manual override: Specific checkpoint paths")
    
    args = parser.parse_args()
    run_evaluation(args)

if __name__ == "__main__":
    main()
