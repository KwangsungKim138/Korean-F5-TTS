"""
Prepare CoreaSpeech dataset for Grapheme mode.
Reads metadata_train.txt where the 3th column (index 2) is the pronunciation text (before jamo decomposition).
Decomposes the text into jamos and saves as raw.arrow, duration.json, vocab.txt.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

import soundfile as sf
from datasets import Dataset as Dataset_
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from f5_tts.model.utils import (
    GRAPHEME_CHOSEONG,
    GRAPHEME_JUNGSEONG,
    GRAPHEME_JONGSEONG,
    convert_char_to_grapheme,
)

def _iter_from_coreaspeech(transcript_path: Path, dataset_dir: Path):
    """Yield (audio_path, text, duration) from CoreaSpeech metadata."""
    with open(transcript_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        rel_path = parts[0]
        # index 2 is the 3rd chunk: n2gk+ processed text (before g2p)
        text = parts[2] 
        wav_path = dataset_dir / rel_path
        if not wav_path.exists():
            continue
        try:
            duration = sf.info(str(wav_path)).duration
        except Exception:
            continue
        if not (0.4 <= duration <= 30):
            continue
        yield str(wav_path), text, duration

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CoreaSpeech -> grapheme (jamo). Output: raw.arrow, duration.json, vocab.txt."
    )
    parser.add_argument("--name", type=str, default="CoreaSpeech_grapheme", help="Output dataset name.")
    parser.add_argument("--data-root", type=Path, default=None, help="Data root (default: project_root/data).")
    parser.add_argument("--transcript", type=Path, required=True, help="CoreaSpeech metadata_train.txt path.")
    parser.add_argument("--wav-dir", type=Path, required=True, help="CoreaSpeech WAV root directory.")
    
    args = parser.parse_args()

    project_root = Path(os.getcwd())
    data_root = args.data_root or (project_root / "data")
    save_dir = data_root / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = args.transcript
    dataset_dir = args.wav_dir
    if not transcript_path.exists():
        print(f"Error: Transcript not found: {transcript_path}")
        return
        
    it = _iter_from_coreaspeech(transcript_path, dataset_dir)
    print(f"Input: transcript {transcript_path}")
    print(f"Output: {save_dir}")
    print("Pipeline: CoreaSpeech n2gk+ Text -> Jamo Decompose\n")

    punctuation_set = set()
    result = []
    duration_list = []

    for audio_path, text_raw, duration in tqdm(it, desc="Processing"):
        # Collect punctuation from text
        for c in text_raw:
            if c != " " and not (ord("가") <= ord(c) <= ord("힣")):
                punctuation_set.add(c)
        
        # Convert to Grapheme (Jamo decompose)
        try:
            grapheme_list = convert_char_to_grapheme([text_raw])[0]
        except Exception as e:
            print(f"Skip convert error for '{text_raw[:50]}...': {e}")
            continue
            
        result.append({"audio_path": audio_path, "text": grapheme_list, "duration": duration})
        duration_list.append(duration)

    if not result:
        print("No samples produced.")
        return

    # Create Vocab: All possible Jamos + Punctuation
    vocab_set = (
        set(GRAPHEME_CHOSEONG)
        | set(GRAPHEME_JUNGSEONG)
        | set(j for j in GRAPHEME_JONGSEONG if j)
        | punctuation_set
    )
    vocab_set.discard("")
        
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
