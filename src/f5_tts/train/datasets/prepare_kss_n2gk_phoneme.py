"""
Prepare dataset with pipeline: N2gk+ -> G2P (Phoneme).
Supports transcript (KSS pipe-separated) or Parquet input.
Output: raw.arrow, duration.json, vocab.txt; optional raw.parquet.
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
    PHONEME_CONSONANTS,
    PHONEME_VOWELS,
    convert_char_to_phoneme,
    convert_char_to_phoneme_skipTC,
)
from f5_tts.train.datasets.normalization_n2gk import normalize_n2gk_plus


def _iter_from_transcript(transcript_path: Path, dataset_dir: Path):
    """Yield (audio_path, text, duration) from KSS-style transcript."""
    with open(transcript_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        rel_path, text = parts[0], parts[2]
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


def _iter_from_parquet(parquet_path: Path, text_col: str, audio_col: str, duration_col: str, audio_base: Path | None):
    """Yield (audio_path, text, duration) from Parquet."""
    ds = Dataset_.from_parquet(str(parquet_path))
    for row in ds:
        text = row.get(text_col) or ""
        ap = row.get(audio_col) or ""
        dur = float(row.get(duration_col, 0))
        if not text or not ap:
            continue
        if audio_base is not None and not os.path.isabs(ap):
            ap = str(audio_base / ap)
        if not (0.4 <= dur <= 30):
            continue
        yield ap, text, dur


def main() -> None:
    parser = argparse.ArgumentParser(
        description="N2gk+ -> phoneme (standard g2p). Output: raw.arrow, duration.json, vocab.txt."
    )
    parser.add_argument("--name", type=str, default="KSS_n2gk_phoneme", help="Output dataset name.")
    parser.add_argument("--data-root", type=Path, default=None, help="Data root (default: project_root/data).")
    # Transcript input (KSS-style)
    parser.add_argument("--transcript", type=Path, default=None, help="KSS transcript path (pipe-separated).")
    parser.add_argument("--kss-dir", type=Path, default=None, help="KSS WAV root (default: data_root/KSS).")
    # Parquet input
    parser.add_argument("--parquet", type=Path, default=None, help="Input Parquet path.")
    parser.add_argument("--text-col", type=str, default="text", help="Text column name in Parquet.")
    parser.add_argument("--audio-col", type=str, default="audio_path", help="Audio path column in Parquet.")
    parser.add_argument("--duration-col", type=str, default="duration", help="Duration column in Parquet.")
    parser.add_argument("--audio-base", type=Path, default=None, help="Base dir for relative paths.")
    # Output
    parser.add_argument("--write-parquet", action="store_true", help="Also write raw.parquet.")
    parser.add_argument("--n2gk-natural", action="store_true", default=True, help="N2gk+ natural=True (default).")
    # SkipTC options
    parser.add_argument("--skip-tc", action="store_true", help="Add SkipTC token at syllable boundary.")
    parser.add_argument("--tokenizer_version", type=str, default=None, help="Legacy skipTC option.")
    
    args = parser.parse_args()

    project_root = Path(os.getcwd())
    data_root = args.data_root or (project_root / "data")
    save_dir = data_root / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.parquet is not None:
        if not args.parquet.exists():
            print(f"Error: Parquet not found: {args.parquet}")
            return
        it = _iter_from_parquet(
            args.parquet,
            args.text_col,
            args.audio_col,
            args.duration_col,
            args.audio_base,
        )
        print(f"Input: Parquet {args.parquet}")
    else:
        transcript_path = args.transcript or (data_root / "KSS" / "transcript.v.1.4.txt")
        dataset_dir = args.kss_dir or (data_root / "KSS")
        if not transcript_path.exists():
            print(f"Error: Transcript not found: {transcript_path}")
            return
        it = _iter_from_transcript(transcript_path, dataset_dir)
        print(f"Input: transcript {transcript_path}")

    use_skip_tc = getattr(args, "skip_tc", False)
    use_legacy = args.tokenizer_version in ("2026-02-07", "legacy") if use_skip_tc else False
    
    print(f"Output: {save_dir}")
    print(f"Pipeline: N2gk+ -> Phoneme (SkipTC={use_skip_tc})\n")

    punctuation_set = set()
    result = []
    duration_list = []

    for audio_path, text_raw, duration in tqdm(it, desc="Processing"):
        # 1. N2gk+ Normalization
        normalized = normalize_n2gk_plus(text_raw, natural=args.n2gk_natural)
        
        # Collect punctuation from normalized text
        for c in normalized:
            if c != " " and not (ord("가") <= ord(c) <= ord("힣")):
                punctuation_set.add(c)
        
        # 2. Convert to Phoneme
        try:
            if use_skip_tc:
                phoneme_list = convert_char_to_phoneme_skipTC([normalized], legacy=use_legacy)[0]
            else:
                phoneme_list = convert_char_to_phoneme([normalized])[0]
        except Exception as e:
            print(f"Skip convert error for '{normalized[:50]}...': {e}")
            continue
            
        result.append({"audio_path": audio_path, "text": phoneme_list, "duration": duration})
        duration_list.append(duration)

    if not result:
        print("No samples produced.")
        return

    # Create Vocab: Phoneme Consonants + Vowels + Punctuation
    vocab_set = (
        set(PHONEME_CONSONANTS) | set(PHONEME_VOWELS) | punctuation_set
    )
    vocab_set.discard("")
    
    if use_skip_tc and not use_legacy:
        vocab_set.add("*")
        
    vocab_list = [" "] + sorted(vocab_set)

    # raw.arrow
    with ArrowWriter(path=str(save_dir / "raw.arrow")) as writer:
        for row in tqdm(result, desc="Writing raw.arrow"):
            writer.write(row)
        writer.finalize()

    if args.write_parquet:
        ds = Dataset_.from_list(result)
        ds.to_parquet(save_dir / "raw.parquet", index=False)
        print(f"Wrote {save_dir / 'raw.parquet'}")

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
