"""
CoreaSpeech / KUB-style test list evaluation (Whisper CER/WER, UTMOS, WavLM SIM).

Test list format (pipe-separated, one utterance per line):
  path | text | duration | speaker

- path: WAV path relative to --data-root
- text: reference transcript for metrics and TTS generation
- duration: seconds (third column); stored on each item for optional use
- speaker: optional fourth column (may be empty); parsed but not used for matching

Reference mapping: consecutive pairs (0↔1), (2↔3), …; if odd count, last item maps to itself.
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
    "grapheme": 355,
    "phoneme": 355,
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
}

# Mode -> Hydra dataset name suffix (ckpts/...CoreaSpeech_<name>_lora)
MODE_DATASET = {
    "grapheme": "CoreaSpeech_grapheme",
    "phoneme": "CoreaSpeech_phoneme",
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
    KUB-style line: path | text | duration | speaker
    - duration: third column (float seconds)
    - speaker: fourth column optional; may be empty; not used for logic
    Also accepts path | text | duration (3 columns, no speaker).
    """
    parts = line.strip().split("|")
    if len(parts) < 3:
        return None
    try:
        item = {
            "path": parts[0].strip(),
            "text": parts[1],
            "duration": float(parts[2]),
        }
        if len(parts) >= 4:
            item["speaker"] = parts[3].strip()
        else:
            item["speaker"] = ""
        if not item["path"]:
            return None
        return item
    except ValueError:
        return None


def build_reference_mapping(test_path: str):
    """
    Pair-based reference: items at 2i and 2i+1 reference each other.
    If odd length, last item references itself.
    """
    print("Building pair-based reference mapping from KUB test list...")
    test_items = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            item = parse_KUB_line(line)
            if item:
                test_items.append(item)

    mapping = {}
    for i in range(0, len(test_items), 2):
        if i + 1 < len(test_items):
            a, b = test_items[i], test_items[i + 1]
            mapping[a["path"]] = b
            mapping[b["path"]] = a
        else:
            mapping[test_items[i]["path"]] = test_items[i]

    return test_items, mapping


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


def run_evaluation(args):
    data_root = os.path.abspath(args.data_root)
    test_path = args.test_txt
    output_base = args.output_dir
    vocab_file = args.vocab

    if not os.path.isfile(test_path):
        print(f"Error: test list not found: {test_path}")
        return

    test_items, ref_mapping = build_reference_mapping(test_path)
    total_items = len(test_items)
    if total_items == 0:
        print("Error: no valid lines in test list.")
        return

    vocoder = None
    whisper_model = None
    utmos_predictor = None
    wavlm_bundle, wavlm_model = None, None

    ref_cache = {}
    final_summary = []

    # --------------------------
    # Ground Truth pass (optional, same as evaluate_models_1h)
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
                refs_gt = df["gt"].fillna("").tolist()
                hyps_gt = df["pred"].fillna("").tolist()
                refs_cer = [r.replace(" ", "") for r in refs_gt]
                hyps_cer = [h.replace(" ", "") for h in hyps_gt]
                final_summary.append(
                    {
                        "Model": "GroundTruth",
                        "Step": "N/A",
                        "CER": min(1.0, jiwer.cer(refs_cer, hyps_cer)) if refs_cer else 0.0,
                        "WER": min(1.0, jiwer.wer(refs_gt, hyps_gt)) if refs_gt else 0.0,
                        "UTMOS": df["utmos"].mean(),
                        "SIM": df["sim"].mean(),
                    }
                )
        except Exception:
            pass

    if not gt_done:
        print("\n" + "=" * 50)
        print("Evaluating Ground Truth (GT)")
        print("=" * 50)
        print(f"[GT] Preparing audio symlinks in {gt_wav_dir}...")
        for item in test_items:
            src_path = resolve_audio_path(data_root, item["path"])
            dst_path = os.path.join(gt_wav_dir, os.path.basename(item["path"]))
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
            test_fname = os.path.basename(item["path"])
            wav_path = os.path.join(gt_wav_dir, test_fname)
            try:
                asr_out = whisper_model.transcribe(wav_path, language="ko", temperature=0.0)
                pred_text = asr_out["text"]
            except Exception as e:
                print(f"Whisper Error: {e}")
                pred_text = ""

            gt_clean = post_process_for_metric(normalize_n2gk_plus(item["text"]))
            pred_clean = post_process_for_metric(normalize_n2gk_plus(pred_text))
            gt_ns = gt_clean.replace(" ", "")
            pred_ns = pred_clean.replace(" ", "")
            cer = min(1.0, jiwer.cer(gt_ns, pred_ns)) if gt_ns else 1.0
            wer = min(1.0, jiwer.wer(gt_clean, pred_clean)) if gt_clean else 1.0
            utmos_score = gt_utmos_results.get(test_fname, np.nan)
            sim_score = np.nan
            ref_item = ref_mapping.get(item["path"])
            if ref_item:
                ref_wav_path = resolve_audio_path(data_root, ref_item["path"])
                if os.path.exists(ref_wav_path) and wavlm_model is not None:
                    sim_score = calculate_sim(wavlm_bundle, wavlm_model, ref_wav_path, wav_path)
            gt_metrics.append(
                {
                    "filename": test_fname,
                    "gt": gt_clean,
                    "pred": pred_clean,
                    "cer": cer,
                    "wer": wer,
                    "utmos": utmos_score,
                    "sim": sim_score,
                }
            )

        df_gt = pd.DataFrame(gt_metrics)
        df_gt.to_csv(gt_details_path, index=False, encoding="utf-8-sig")
        refs_gt = df_gt["gt"].fillna("").tolist()
        hyps_gt = df_gt["pred"].fillna("").tolist()
        refs_cer = [r.replace(" ", "") for r in refs_gt]
        hyps_cer = [h.replace(" ", "") for h in hyps_gt]
        mean_cer = min(1.0, jiwer.cer(refs_cer, hyps_cer)) if refs_cer else 0.0
        mean_wer = min(1.0, jiwer.wer(refs_gt, hyps_gt)) if refs_gt else 0.0
        mean_utmos = df_gt["utmos"].mean()
        mean_sim = df_gt["sim"].mean()
        print(f"[GT] Result -> CER: {mean_cer:.4f}, WER: {mean_wer:.4f}, UTMOS: {mean_utmos:.4f}, SIM: {mean_sim:.4f}")
        final_summary.append(
            {
                "Model": "GroundTruth",
                "Step": "N/A",
                "CER": mean_cer,
                "WER": mean_wer,
                "UTMOS": mean_utmos,
                "SIM": mean_sim,
            }
        )

    # --------------------------
    # Model evaluation
    # --------------------------
    for mode, best_step in BEST_MODELS.items():
        dataset_name = MODE_DATASET.get(mode, f"CoreaSpeech_{MODE_MAP.get(mode, mode)}")
        ckpt_dir = find_ckpt_dir(mode, dataset_name)
        if not ckpt_dir:
            ckpt_dir = os.path.join("ckpts", f"F5TTS_Base_vocos_custom_{dataset_name}_lora")
            print(f"[{mode}] Checkpoint dir not found; expected under ckpts/*{dataset_name}*_lora — using placeholder: {ckpt_dir}")

        tokenizer_name = mode_to_tokenizer(mode)
        config_path = default_config_path(mode, dataset_name)

        print("\n" + "=" * 50)
        print(f"Evaluating Mode: {mode} (Dir: {ckpt_dir})")
        print("=" * 50)

        for step in [best_step]:
            model_id = f"{mode}_{step}K"
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
                except Exception:
                    pass

            if is_result_complete:
                print(f"[{model_id}] Results complete. Skipping.")
                try:
                    df = pd.read_csv(details_csv_path)
                    refs = df["gt"].fillna("").tolist()
                    hyps = df["pred"].fillna("").tolist()
                    refs_cer = [r.replace(" ", "") for r in refs]
                    hyps_cer = [h.replace(" ", "") for h in hyps]
                    final_summary.append(
                        {
                            "Model": mode,
                            "Step": step,
                            "CER": min(1.0, jiwer.cer(refs_cer, hyps_cer)) if refs_cer else 0.0,
                            "WER": min(1.0, jiwer.wer(refs, hyps)) if refs else 0.0,
                            "UTMOS": df["utmos"].mean(),
                            "SIM": df["sim"].mean(),
                        }
                    )
                except Exception:
                    pass
                continue

            if not is_generation_complete:
                print(f"[{model_id}] Generating ({len(existing_wavs)}/{total_items})...")
                ckpt_filename = f"model_{step}000.pt"
                ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
                if not os.path.isfile(ckpt_path):
                    nested = []
                    for root, _dirs, files in os.walk(ckpt_dir):
                        if ckpt_filename in files:
                            nested.append(os.path.join(root, ckpt_filename))
                    if nested:
                        ckpt_path = sorted(nested)[-1]
                    else:
                        print(f"[{model_id}] Checkpoint not found: {ckpt_filename} under {ckpt_dir}")
                        continue

                try:
                    if vocoder is None:
                        vocoder = load_vocoder_model()
                    if config_path and os.path.isfile(config_path):
                        conf = OmegaConf.load(config_path)
                        model_arch_config = conf.model.arch
                        model_cls_name = conf.model.backbone
                    else:
                        model_arch_config = dict(
                            dim=1024,
                            depth=22,
                            heads=16,
                            ff_mult=2,
                            text_dim=512,
                            conv_layers=4,
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
                    test_fname = os.path.basename(item["path"])
                    output_wav_path = os.path.join(output_dir, test_fname)
                    if os.path.exists(output_wav_path):
                        continue

                    ref_item = ref_mapping.get(item["path"])
                    ref_audio, ref_text = None, None
                    if ref_item:
                        if ref_item["path"] in ref_cache:
                            ref_audio, ref_text = ref_cache[ref_item["path"]]
                        else:
                            try:
                                ref_path_full = resolve_audio_path(data_root, ref_item["path"])
                                ref_audio, ref_text = preprocess_ref_audio_text(
                                    ref_path_full, ref_item["text"], show_info=lambda x: None
                                )
                                ref_cache[ref_item["path"]] = (ref_audio, ref_text)
                            except Exception:
                                pass
                    if ref_audio is None:
                        try:
                            fb = resolve_audio_path(data_root, test_items[0]["path"])
                            ref_audio, ref_text = preprocess_ref_audio_text(
                                fb, test_items[0]["text"], show_info=lambda x: None
                            )
                        except Exception:
                            pass

                    try:
                        audio, sr, _ = infer_process(
                            ref_audio,
                            ref_text,
                            item["text"],
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
                print(f"[{model_id}] Generation complete. Evaluating.")

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
                test_fname = os.path.basename(item["path"])
                output_wav_path = os.path.join(output_dir, test_fname)
                if not os.path.exists(output_wav_path):
                    continue
                try:
                    asr_out = whisper_model.transcribe(output_wav_path, language="ko", temperature=0.0)
                    pred_text = asr_out["text"]
                except Exception as e:
                    print(f"Whisper Error: {e}")
                    pred_text = ""

                gt_clean = post_process_for_metric(normalize_n2gk_plus(item["text"]))
                pred_clean = post_process_for_metric(normalize_n2gk_plus(pred_text))
                gt_ns = gt_clean.replace(" ", "")
                pred_ns = pred_clean.replace(" ", "")
                cer = min(1.0, jiwer.cer(gt_ns, pred_ns)) if gt_ns else 1.0
                wer = min(1.0, jiwer.wer(gt_clean, pred_clean)) if gt_clean else 1.0
                utmos_score = utmos_results.get(test_fname, np.nan)
                sim_score = np.nan
                if actual_run_sim:
                    ref_item = ref_mapping.get(item["path"])
                    if ref_item:
                        ref_wav_path = resolve_audio_path(data_root, ref_item["path"])
                        if os.path.exists(ref_wav_path):
                            sim_score = calculate_sim(wavlm_bundle, wavlm_model, ref_wav_path, output_wav_path)
                eval_metrics.append(
                    {
                        "filename": test_fname,
                        "gt": gt_clean,
                        "pred": pred_clean,
                        "cer": cer,
                        "wer": wer,
                        "utmos": utmos_score,
                        "sim": sim_score,
                    }
                )

            df_res = pd.DataFrame(eval_metrics)
            df_res.to_csv(details_csv_path, index=False, encoding="utf-8-sig")
            refs = df_res["gt"].fillna("").tolist()
            hyps = df_res["pred"].fillna("").tolist()
            refs_cer = [r.replace(" ", "") for r in refs]
            hyps_cer = [h.replace(" ", "") for h in hyps]
            mean_cer = min(1.0, jiwer.cer(refs_cer, hyps_cer)) if refs_cer else 0.0
            mean_wer = min(1.0, jiwer.wer(refs, hyps)) if refs else 0.0
            mean_utmos = df_res["utmos"].mean()
            mean_sim = df_res["sim"].mean()
            print(f"[{model_id}] Result -> CER: {mean_cer:.4f}, WER: {mean_wer:.4f}, UTMOS: {mean_utmos:.4f}, SIM: {mean_sim:.4f}")
            final_summary.append(
                {
                    "Model": mode,
                    "Step": step,
                    "CER": mean_cer,
                    "WER": mean_wer,
                    "UTMOS": mean_utmos if actual_run_utmos else "",
                    "SIM": mean_sim if actual_run_sim else "",
                }
            )

    if final_summary:
        df_summary = pd.DataFrame(final_summary)
        print("\n" + "=" * 80)
        print("FINAL EVALUATION REPORT (CoreaSpeech KUB)")
        print("=" * 80)
        print(df_summary.to_string(index=False))
        save_path = os.path.join(output_base, "evaluation_summary_comprehensive.csv")
        df_summary.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\nReport saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate F5-TTS on CoreaSpeech / KUB test list")
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Root directory; first column paths are relative to this",
    )
    parser.add_argument(
        "--test-txt",
        type=str,
        default=DEFAULT_TEST_TXT,
        help="Test list: path|text|duration|speaker",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for wavs, details.csv, summary",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=DEFAULT_VOCAB,
        help="vocab.txt for load_model (custom vocab path)",
    )
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
