import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from f5_tts.model.utils import (
    convert_char_to_phoneme,
    convert_char_to_phoneme_skipTC,
    PHONEME_CONSONANTS,
    PHONEME_VOWELS,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare KSS metadata with phoneme tokenizer -> raw.arrow, duration.json, vocab.txt"
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Path to metadata file (default: data/KSS/transcript.v.1.4.txt). Can use train_1h.txt etc.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="KSS",
        help="Dataset output name. Saves to data/{name}_kor_phoneme (e.g. KSS_1h, KSS_3h for modes).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data root directory (default: project_root/data). WAV paths resolved under {data-root}/KSS/.",
    )
    parser.add_argument(
        "--skip-tc",
        action="store_true",
        help="Add SkipTC token at syllable boundary (empty coda). Default: no skipTC.",
    )
    args = parser.parse_args()

    project_root = Path(os.getcwd())
    data_root = args.data_root or (project_root / "data")
    dataset_dir = data_root / "KSS"
    transcript_path = args.transcript or (dataset_dir / "transcript.v.1.4.txt")
    dataset_name = args.name
    tokenizer_type = "kor_phoneme"
    save_dir = data_root / f"{dataset_name}_{tokenizer_type}"

    use_skip_tc = getattr(args, "skip_tc", False)
    if use_skip_tc:
        tok_ver_str = "skipTC=*"
    else:
        tok_ver_str = "no skipTC (default)"
    print(f"\nPrepare for {dataset_name} with {tokenizer_type} tokenizer")
    print(f"Mode: {tok_ver_str}")
    print(f"Transcript: {transcript_path}")
    print(f"WAV base dir: {dataset_dir}")
    print(f"Saving to: {save_dir}\n")

    if not transcript_path.exists():
        print(f"Error: Transcript file not found at {transcript_path}")
        return

    with open(transcript_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 데이터셋에 등장하는 구두점/기호 수집 (한글·공백 제외)
    punctuation_set = set()
    for line in lines:
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        for char in parts[2]:
            if char != " " and not (ord("가") <= ord(char) <= ord("힣")):
                punctuation_set.add(char)

    # Phoneme vocab = 음소 자음 + 음소 모음 + 데이터셋 구두점, 빈 문자 '' 제외
    vocab_set = (
        set(PHONEME_CONSONANTS) | set(PHONEME_VOWELS) | punctuation_set
    )
    vocab_set.discard("")
    if use_skip_tc:
        vocab_set.add("*")

    result = []
    duration_list = []

    print("Processing audio and text...")

    for line in tqdm(lines):
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
            
        rel_path = parts[0]
        text_content = parts[2] # Use expanded text
        
        wav_path = dataset_dir / rel_path
        
        if not wav_path.exists():
            print(f"Warning: Audio file not found: {wav_path}")
            continue

        # Get duration
        try:
            duration = sf.info(str(wav_path)).duration
        except Exception as e:
            print(f"Error reading audio {wav_path}: {e}")
            continue

        if duration < 0.4 or duration > 30:
            continue

        # Convert to phonemes
        try:
            if use_skip_tc:
                phoneme_tokens = convert_char_to_phoneme_skipTC([text_content])[0]
            else:
                phoneme_tokens = convert_char_to_phoneme([text_content])[0]
        except Exception as e:
            print(f"Error converting text '{text_content}': {e}")
            continue
            
        result.append({
            "audio_path": str(wav_path),
            "text": phoneme_tokens,
            "duration": duration
        })
        duration_list.append(duration)

    # Create output directory
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Save raw.arrow
    print(f"\nWriting {len(result)} samples to raw.arrow ...")
    with ArrowWriter(path=str(save_dir / "raw.arrow")) as writer:
        for line in tqdm(result, desc="Writing"):
            writer.write(line)
        writer.finalize()

    # Save duration.json
    import json
    with open(save_dir / "duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # Save vocab.txt (고정 집합: 음소 자음+모음+구두점, 빈 문자 '' 제외)
    print("\nGenerating vocab.txt ...")
    vocab_list = [" "] + sorted(vocab_set)
    with open(save_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for v in vocab_list:
            f.write(v + "\n")

    print(f"Vocab size: {len(vocab_list)}")
    print(f"Total duration: {sum(duration_list) / 3600:.2f} hours")
    print("Done!")

if __name__ == "__main__":
    main()
