import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from f5_tts.model.utils import (
    convert_char_to_allophone,
    PHONEME_CONSONANTS,
    PHONEME_VOWELS,
    PHONEMES_I,
    PHONEMES_P,
    PHONEMES_C,
    MARK_INIT,
    MARK_PAL,
    MARK_CODA,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare KSS metadata with allophone tokenizer -> raw.arrow, duration.json, vocab.txt"
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
        help="Dataset output name. Saves to data/{name}_kor_allophone (e.g. KSS_1h, KSS_3h for modes).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data root directory (default: project_root/data). WAV paths resolved under {data-root}/KSS/.",
    )
    args = parser.parse_args()

    # Paths
    project_root = Path(os.getcwd())
    data_root = args.data_root or (project_root / "data")
    dataset_dir = data_root / "KSS"
    transcript_path = args.transcript or (dataset_dir / "transcript.v.1.4.txt")
    dataset_name = args.name
    tokenizer_type = "kor_allophone"
    save_dir = data_root / f"{dataset_name}_{tokenizer_type}"

    print(f"\nPrepare for {dataset_name} with {tokenizer_type} tokenizer")
    print(f"Transcript: {transcript_path}")
    print(f"WAV base dir: {dataset_dir}")
    print(f"Saving to: {save_dir}\n")

    if not transcript_path.exists():
        print(f"Error: Transcript file not found at {transcript_path}")
        return

    # Read transcript
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

    # Allophone vocab = phoneme vocab(자음+모음+구두점, '' 제외) + 변이음 토큰
    phoneme_base = (
        set(PHONEME_CONSONANTS) | set(PHONEME_VOWELS) | punctuation_set
    )
    phoneme_base.discard("")
    allophone_tokens = (
        {c + MARK_INIT for c in PHONEMES_I}
        | {p + MARK_PAL for p in PHONEMES_P}
        | {c + MARK_CODA for c in PHONEMES_C}
    )
    vocab_set = phoneme_base | allophone_tokens

    result = []
    duration_list = []

    # Process each line
    # Format: relative_path|original|expanded|decomposed|duration|english
    
    print("Processing audio and text...")
    
    # Pre-collect texts for batch processing if needed, 
    # but convert_char_to_allophone takes a list, so we can process in chunks or one by one.
    # For progress bar, let's process one by one or small batches.
    
    # Actually, let's just loop and process.
    for line in tqdm(lines):
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
            
        rel_path = parts[0]
        # Use expanded text (field 2)
        text_content = parts[2]
        
        wav_path = dataset_dir / rel_path
        
        if not wav_path.exists():
            print(f"Warning: Audio file not found: {wav_path}")
            continue

        # Get duration
        try:
            # KSS transcript has duration in field 4, but let's verify with soundfile just in case
            # or trust the transcript to be faster? 
            # Let's use soundfile to be safe and consistent with other scripts
            duration = sf.info(str(wav_path)).duration
        except Exception as e:
            print(f"Error reading audio {wav_path}: {e}")
            continue

        # Filter by duration (standard practice)
        if duration < 0.4 or duration > 30:
            continue

        # Convert text to allophones
        # convert_char_to_allophone expects a list of strings and returns a list of list of tokens
        try:
            allophone_tokens_list = convert_char_to_allophone([text_content])
            allophone_tokens = allophone_tokens_list[0]  # Get the first (and only) result

        except Exception as e:
            print(f"Error converting text '{text_content}': {e}")
            continue
            
        result.append({
            "audio_path": str(wav_path),
            "text": allophone_tokens, # List of strings
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

    # Save vocab.txt (phoneme 집합 + 변이음, 빈 문자 '' 제외)
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
