import os
import random
import re
import argparse
import soundfile as sf
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import get_class

# F5-TTS 추론 유틸리티 로드
from f5_tts.infer.utils_infer import (
    load_model, load_vocoder, infer_process, preprocess_ref_audio_text, device
)

def main():
    parser = argparse.ArgumentParser(description="CoreaSpeech Validation Inference")
    parser.add_argument("--ckpt", type=str, required=True, help="테스트할 체크포인트 경로 (.pt)")
    parser.add_argument("--num_samples", type=int, default=10, help="생성할 샘플 개수")
    args = parser.parse_args()

    # 경로 설정
    data_root = os.path.expanduser("~/datasets/CoreaSpeech/coreaspeech_salt")
    valid_txt = os.path.join(data_root, "metadata_valid.txt")
    
    config_path = "src/f5_tts/configs/F5TTS_Base_ft_Lora_RTX3090_CoreaSpeech_grapheme.yaml"
    vocab_path = "data/CoreaSpeech_grapheme/vocab.txt"
    output_dir = "eval_results/coreaspeech_valid"
    os.makedirs(output_dir, exist_ok=True)

    print("1. 검증셋 메타데이터 로딩 중...")
    valid_data = []
    with open(valid_txt, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                wav_path = os.path.join(data_root, parts[0]) # 예: wav/파일명.wav
                raw_text = parts[1] # 인덱스 1: 원본 한국어 텍스트
                if os.path.exists(wav_path):
                    valid_data.append((wav_path, raw_text))
    
    print(f" -> 총 {len(valid_data)}개의 유효한 검증 데이터를 찾았습니다.")

    print("\n2. 모델 및 보코더 로딩 중...")
    model_cfg = OmegaConf.load(config_path)
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    vocoder = load_vocoder(model_cfg.model.mel_spec.mel_spec_type, device=device)
    
    # 모델 로드 (내부적으로 vocab을 읽어 자동으로 Grapheme 모드로 작동함)
    model = load_model(
        model_cls,
        model_cfg.model.arch,
        args.ckpt,
        mel_spec_type=model_cfg.model.mel_spec.mel_spec_type,
        vocab_file=vocab_path,
        device=device,
        tokenizer="custom",
        use_n2gk_plus=True
    )

    # 평가할 샘플 무작위 추출
    eval_samples = random.sample(valid_data, min(args.num_samples, len(valid_data)))
    
    print(f"\n3. {len(eval_samples)}개 문장에 대한 추론 시작...")
    for i, (target_wav, target_text) in enumerate(eval_samples):
        # 목소리 복제의 기준(Reference)이 될 오디오를 검증셋에서 무작위로 하나 선택
        ref_wav, ref_text = random.choice(valid_data)
        
        print(f"[{i+1}/{len(eval_samples)}] 생성 중: {target_text[:30]}...")
        
        # 레퍼런스 오디오 전처리
        ref_audio_pre, ref_text_pre = preprocess_ref_audio_text(ref_wav, ref_text, show_info=lambda x: None)
        
        # 추론 진행
        result = infer_process(
            ref_audio_pre,
            ref_text_pre,
            target_text,
            model,
            vocoder,
            mel_spec_type=model_cfg.model.mel_spec.mel_spec_type,
            speed=1.0,
            device=device,
            show_info=lambda x: None
        )
        
        # 결과 저장
        if result:
            final_wave, final_sample_rate, _ = result
            # 파일명에 특수문자 제거 및 공백 치환
            safe_text = re.sub(r'[^\w\s]', '', target_text).replace(" ", "_")[:20]
            out_path = os.path.join(output_dir, f"sample_{i+1:02d}_{safe_text}.wav")
            sf.write(out_path, final_wave, final_sample_rate)
            print(f"  -> 저장 완료: {out_path}")

    print("\n✅ 모든 추론이 완료되었습니다!")

if __name__ == "__main__":
    main()