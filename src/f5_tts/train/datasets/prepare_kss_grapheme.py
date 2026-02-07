import os
import sys
import shutil
from pathlib import Path

sys.path.append(os.getcwd())

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from f5_tts.model.utils import (
    _syllable_to_phonemes,
    PHONEME_CONSONANTS,
    PHONEME_VOWELS
)

def generate_grapheme_vocab():
    vocab = []
    # 0. Space (Padding/Separator)
    vocab.append(" ")
    
    # 1. Jamos (same as basic phonemes, but treating them as graphemes)
    for consonant in PHONEME_CONSONANTS:
        if consonant not in vocab:
            vocab.append(consonant)
    for vowel in PHONEME_VOWELS:
        if vowel not in vocab:
            vocab.append(vowel)
            
    # Add punctuation
    punctuation = [".", ",", "!", "?", "~", "…"]
    for p in punctuation:
        if p not in vocab:
            vocab.append(p)

    return vocab

def convert_text_to_jamos(text: str) -> list[str]:
    """
    Convert text to Jamos (Graphemes) simply by decomposing Hangul.
    NO G2P (Pronunciation Rule) applied.
    """
    jamos = []
    for char in text:
        if char == ' ':
            jamos.append(' ')
        else:
            # Decompose syllable using the utility
            # This handles Hangul decomposition, and leaves other chars as is
            decomposed = _syllable_to_phonemes(char)
            jamos.extend(decomposed)
            
    return jamos

def main():
    # Configuration
    dataset_name = "KSS"
    tokenizer_type = "kor_grapheme"
    
    # Paths
    project_root = Path(os.getcwd())
    data_root = project_root / "data"
    dataset_dir = data_root / "KSS"
    transcript_path = dataset_dir / "transcript.v.1.4.txt"
    
    # Output directory: data/KSS_grapheme
    save_dir = data_root / f"{dataset_name}_{tokenizer_type}"
    
    print(f"\nPrepare for {dataset_name} with {tokenizer_type} tokenizer")
    print(f"Reading from: {dataset_dir}")
    print(f"Saving to: {save_dir}\n")

    if not transcript_path.exists():
        print(f"Error: Transcript file not found at {transcript_path}")
        return

    # Read transcript
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
            grapheme_tokens = convert_text_to_jamos(text_content)
            # Update vocab set
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
