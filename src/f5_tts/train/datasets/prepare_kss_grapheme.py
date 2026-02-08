import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from f5_tts.model.utils import convert_char_to_grapheme, convert_char_to_grapheme_skipTC

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare KSS metadata with grapheme (jamo) tokenizer -> raw.arrow, duration.json, vocab.txt"
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
        help="Dataset output name. Saves to data/{name}_kor_grapheme (e.g. KSS_1h, KSS_3h for modes).",
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
    parser.add_argument(
        "--tokenizer_version",
        type=str,
        default=None,
        help="Only if --skip-tc: use '2026-02-07' or 'legacy' for skipTC as ''; omit for '*'.",
    )
    args = parser.parse_args()

    project_root = Path(os.getcwd())
    data_root = args.data_root or (project_root / "data")
    dataset_dir = data_root / "KSS"
    transcript_path = args.transcript or (dataset_dir / "transcript.v.1.4.txt")
    dataset_name = args.name
    tokenizer_type = "kor_grapheme"
    save_dir = data_root / f"{dataset_name}_{tokenizer_type}"

    use_skip_tc = getattr(args, "skip_tc", False)
    use_legacy = args.tokenizer_version in ("2026-02-07", "legacy") if use_skip_tc else False
    if use_skip_tc:
        tok_ver_str = "legacy (2026-02-07, skipTC='')" if use_legacy else "skipTC=*"
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

    result = []
    duration_list = []
    vocab_set = set()
    vocab_set.add(" ")
    
    print("Processing audio and text (Grapheme Baseline)...")
    
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

        try:
            duration = sf.info(str(wav_path)).duration
        except Exception as e:
            print(f"Error reading audio {wav_path}: {e}")
            continue

        if duration < 0.4 or duration > 30:
            continue

        # Convert to Jamos (Graphemes)
        try:
            if use_skip_tc:
                grapheme_tokens = convert_char_to_grapheme_skipTC([text_content], legacy=use_legacy)[0]
            else:
                grapheme_tokens = convert_char_to_grapheme([text_content])[0]
            vocab_set.update(grapheme_tokens)
        except Exception as e:
            print(f"Error converting text '{text_content}': {e}")
            continue
            
        result.append({
            "audio_path": str(wav_path),
            "text": grapheme_tokens,
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

    # Save vocab.txt
    print("\nGenerating vocab.txt ...")
    
    with open(save_dir / "vocab.txt", "w", encoding="utf-8") as f:
        if " " in vocab_set:
            vocab_set.remove(" ")
        f.write(" \n")
        for v in sorted(list(vocab_set)):
            f.write(v + "\n")

    print(f"Vocab size: {len(vocab_set) + 1}")
    print(f"Total duration: {sum(duration_list) / 3600:.2f} hours")
    print("Done!")

if __name__ == "__main__":
    main()
