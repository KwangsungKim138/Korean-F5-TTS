import os
import argparse
from importlib.resources import files
from pathlib import Path
import soundfile as sf
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import get_class


from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    infer_process,
    preprocess_ref_audio_text,
)


CKPT_FILE = "ckpts/F5TTS_0209/model_450K.pt"
VOCAB_FILE = "data/KSS_full_kor_allophone/vocab.txt"
REF_AUDIO_PATH = "data/KSS/1/1_0001.wav"
REF_TEXT_CONTENT = "그는 괜찮은 척하려고 애쓰는 것 같았다."
OUTPUT_DIR = "eval/testaudio_450K"
TRANSCRIPT_PATH = "/home/waegari/projects/F5-TTS/data/KSS/test_for_eval.txt"
MODEL_NAME = "F5TTS_Small"  # Bash 예시에 명시된 모델명
DEVICE = "cuda"


VOCODER_NAME = "vocos"
USE_N2GK_PLUS = True  # N2gk+ before g2p/allophone (set False for models trained without N2gk+)
TARGET_RMS = 0.1
CROSS_FADE_DURATION = 0.15
NFE_STEP = 32
CFG_STRENGTH = 2.0
SWAY_SAMPLING_COEF = -1.0
SPEED = 1.0

def main():
    # 1. 출력 디렉토리 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. Vocoder 로드
    print(f"Loading Vocoder ({VOCODER_NAME})...")
    vocoder = load_vocoder(vocoder_name=VOCODER_NAME, is_local=False, device=DEVICE)

    # 3. Model 로드
    print(f"Loading Model ({MODEL_NAME}) from {CKPT_FILE}...")
    try:
        # config 파일 경로
        model_cfg_path = str(files("f5_tts").joinpath(f"configs/{MODEL_NAME}.yaml"))
        model_cfg = OmegaConf.load(model_cfg_path)
    except Exception as e:
        print(f"Error loading config for {MODEL_NAME}: {e}")
        return

    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    ema_model = load_model(
        model_cls,
        model_arc,
        CKPT_FILE,
        mel_spec_type=VOCODER_NAME,
        vocab_file=VOCAB_FILE,
        device=DEVICE,
        use_n2gk_plus=USE_N2GK_PLUS,
    )

    # 4. Reference Audio 전처리
    print("Processing Reference Audio...")
    ref_audio, ref_text = preprocess_ref_audio_text(REF_AUDIO_PATH, REF_TEXT_CONTENT)

    # 5. Transcript 파일 읽기
    print(f"Reading transcript from {TRANSCRIPT_PATH}...")
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    count = 0
    max_count = 100

    for line in lines:
        if count >= max_count:
            break
        
        parts = line.strip().split("|")
        if len(parts) < 1:
            continue

        # KSS 전용
        filename_base = parts[0].split('/')[-1]
        gen_text = parts[2] 
        
        # 파일명에 확장자가 없으면 .wav 추가
        if not filename_base.endswith(".wav"):
            output_filename = f"{filename_base}.wav"
        else:
            output_filename = filename_base

        print(f"[{count+1}/{max_count}] Generating: {output_filename}")
        print(f" - Text: {gen_text}")

        # 6. Inference
        try:
            audio_segment, final_sample_rate, _ = infer_process(
                ref_audio,
                ref_text,
                gen_text,
                ema_model,
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

            # 7. 파일 저장
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            sf.write(output_path, audio_segment, final_sample_rate)
            
            count += 1

        except Exception as e:
            print(f"Failed to generate {output_filename}: {e}")

    print("Batch inference completed.")

if __name__ == "__main__":
    main()