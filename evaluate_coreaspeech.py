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
    vocab_path = "ckpts/pretrained/vocab_pretr.txt" 
    
    output_dir = "eval_results/coreaspeech_valid"
    os.makedirs(output_dir, exist_ok=True)

    print("1. 검증셋 메타데이터 로딩 중...")
    valid_data = []
    with open(valid_txt, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 5: # duration이 있는 5번째 열까지 확인
                wav_path = os.path.join(data_root, parts[0]) # 예: wav/파일명.wav
                raw_text = parts[1] # 인덱스 1: 원본 한국어 텍스트
                speaker_id = parts[3] # 인덱스 3: 화자 ID
                
                # JSON 형태의 5번째 열 파싱 대신, 실제 오디오 파일을 읽어서 길이를 확인하거나
                # metadata_train.txt처럼 duration 정보가 있다면 그걸 쓰면 좋지만,
                # 현재 metadata_valid.txt 구조상 sf.info로 직접 길이를 확인합니다.
                if os.path.exists(wav_path):
                    try:
                        duration = sf.info(wav_path).duration
                        # 10초 ~ 15초 사이의 오디오만 필터링
                        if 10.0 <= duration <= 15.0:
                            valid_data.append((wav_path, raw_text, speaker_id, duration))
                    except Exception:
                        pass
    
    print(f" -> 총 {len(valid_data)}개의 유효한 검증 데이터를 찾았습니다.")

    # 레퍼런스 매칭 (동일 화자 매칭: 같은 화자의 다른 문장을 레퍼런스로 할당)
    speaker_dict = {}
    for item in valid_data:
        wav_path = item[0]
        # 메타데이터의 4번째 열(index 3)이 화자 ID (예: N35/99C9_GCVR8U39B9_B3N)
        speaker_id = item[2] 
        
        if speaker_id not in speaker_dict:
            speaker_dict[speaker_id] = []
        speaker_dict[speaker_id].append(item)

    reference_mapping = {}
    for item in valid_data:
        wav_path = item[0]
        speaker_id = item[2]
        
        speaker_items = speaker_dict[speaker_id]
        
        # 화자의 문장이 2개 이상이면 자기 자신이 아닌 다른 문장을 선택
        if len(speaker_items) > 1:
            candidates = [x for x in speaker_items if x[0] != wav_path]
            # 재현성을 위해 리스트의 첫 번째 항목 선택
            reference_mapping[wav_path] = candidates[0] 
        else:
            # 화자의 문장이 1개뿐이면 어쩔 수 없이 자기 자신을 레퍼런스로 사용
            reference_mapping[wav_path] = item

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
        tokenizer="kor_grapheme",
        use_n2gk_plus=True
    )

    # 평가할 샘플 무작위 추출
    if len(valid_data) < args.num_samples:
        print(f"⚠️ 경고: 10~15초 조건을 만족하는 데이터가 {len(valid_data)}개뿐입니다. 모두 평가합니다.")
    eval_samples = random.sample(valid_data, min(args.num_samples, len(valid_data)))
    
    print(f"\n3. {len(eval_samples)}개 문장에 대한 추론 시작...")
    for i, (target_wav, target_text, target_speaker, target_duration) in enumerate(eval_samples):
        # 동일 화자로 매칭된 레퍼런스 오디오 가져오기
        ref_wav, ref_text, ref_speaker, ref_duration = reference_mapping[target_wav]
        
        print(f"[{i+1}/{len(eval_samples)}] 생성 중: {target_text[:30]}... (Ref: {ref_duration:.1f}s, Target: {target_duration:.1f}s)")
        
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