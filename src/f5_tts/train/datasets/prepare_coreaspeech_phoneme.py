"""
Prepare CoreaSpeech dataset for Phoneme mode.
Reads metadata_train.txt where the 4th column (index 3) is the pronunciation text.
Converts the text into phonemes and saves as raw.arrow, duration.json, vocab.txt.
Uses multiprocessing to speed up audio header reading and text processing.
"""
import argparse
import json
import os
import sys
from pathlib import Path
import concurrent.futures

sys.path.insert(0, os.getcwd())

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from f5_tts.model.utils import (
    PHONEME_CONSONANTS,
    PHONEME_VOWELS,
    convert_char_to_phoneme,
    convert_char_to_phoneme_skipTC,
)

def process_line(line, dataset_dir, use_skip_tc):
    parts = line.strip().split("|")
    if len(parts) < 4:
        return None
    
    rel_path = parts[0]
    text_raw = parts[3] 
    wav_path = dataset_dir / rel_path
    
    if not wav_path.exists():
        return None
        
    # 1. Read audio duration
    try:
        duration = sf.info(str(wav_path)).duration
    except Exception:
        return None
        
    if not (0.4 <= duration <= 30):
        return None

    # 2. Collect punctuation
    punctuation = set()
    for c in text_raw:
        if c != " " and not (ord("가") <= ord(c) <= ord("힣")):
            punctuation.add(c)
            
    # 3. Convert to Phoneme
    try:
        if use_skip_tc:
            phoneme_list = convert_char_to_phoneme_skipTC([text_raw])[0]
        else:
            phoneme_list = convert_char_to_phoneme([text_raw])[0]
    except Exception:
        return None
        
    return {
        "audio_path": str(wav_path),
        "text": phoneme_list,
        "duration": duration,
        "punctuation": punctuation
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CoreaSpeech -> phoneme. Output: raw.arrow, duration.json, vocab.txt."
    )
    parser.add_argument("--name", type=str, default="CoreaSpeech_phoneme", help="Output dataset name.")
    parser.add_argument("--data-root", type=Path, default=None, help="Data root (default: project_root/data).")
    parser.add_argument("--transcript", type=Path, required=True, help="CoreaSpeech metadata_train.txt path.")
    parser.add_argument("--wav-dir", type=Path, required=True, help="CoreaSpeech WAV root directory.")
    parser.add_argument("--skip-tc", action="store_true", help="Add SkipTC token at syllable boundary (empty coda). Default: no skipTC.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of multiprocessing workers.")
    
    args = parser.parse_args()

    project_root = Path(os.getcwd())
    data_root = args.data_root or (project_root / "data")
    save_dir = data_root / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = args.transcript
    dataset_dir = args.wav_dir
    use_skip_tc = args.skip_tc
    num_workers = args.workers

    if not transcript_path.exists():
        print(f"Error: Transcript not found: {transcript_path}")
        return
        
    print(f"Input: transcript {transcript_path}")
    print(f"Output: {save_dir}")
    print(f"Pipeline: CoreaSpeech Pronunciation Text -> Phoneme (skipTC={use_skip_tc})")
    print(f"Using {num_workers} workers for parallel processing...\n")

    with open(transcript_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    punctuation_set = set()
    result = []
    duration_list = []

    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_line, line, dataset_dir, use_skip_tc) for line in lines]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            res = future.result()
            if res is not None:
                result.append({
                    "audio_path": res["audio_path"],
                    "text": res["text"],
                    "duration": res["duration"]
                })
                duration_list.append(res["duration"])
                punctuation_set.update(res["punctuation"])

    if not result:
        print("No samples produced.")
        return

    # Create Vocab: All possible Phonemes + Punctuation
    vocab_set = (
        set(PHONEME_CONSONANTS)
        | set(PHONEME_VOWELS)
        | punctuation_set
    )
    vocab_set.discard("")
    if use_skip_tc:
        vocab_set.add("*")
        
    vocab_list = [" "] + sorted(vocab_set)

    # raw.arrow
    with ArrowWriter(path=str(save_dir / "raw.arrow")) as writer:
        for row in tqdm(result, desc="Writing raw.arrow"):
            writer.write(row)
        writer.finalize()

    with open(save_dir / "duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    with open(save_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for v in vocab_list:
            f.write(v + "\n")

    print(f"Vocab size: {len(vocab_list)}")
    print(f"Samples: {len(result)}, duration: {sum(duration_list) / 3600:.2f} h")
    print("Done.")

if __name__ == "__main__":
    main()
