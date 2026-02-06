import os
import sys
import shutil
from pathlib import Path

sys.path.append(os.getcwd())

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from f5_tts.model.utils import (
    _text_to_pronunciation,
    _pronunciation_to_eojeols,
    _syllable_to_phonemes,
    PHONEME_CONSONANTS,
    PHONEME_VOWELS
)

def generate_phoneme_vocab():
    vocab = []
    # 0. Space (Padding/Separator)
    vocab.append(" ")
    
    # 1. Basic Phonemes
    for consonant in PHONEME_CONSONANTS:
        if consonant not in vocab:
            vocab.append(consonant)
    for vowel in PHONEME_VOWELS:
        if vowel not in vocab:
            vocab.append(vowel)
            
    # Standard phoneme set usually doesn't include allophone marks
    
    # Add punctuation if needed? 
    # For fair comparison with user's allophone setup (which didn't seem to explicit include punct), 
    # we'll stick to this. But often punctuation is useful.
    # Let's add standard punctuation just in case, as they might be preserved by g2p.
    punctuation = [".", ",", "!", "?", "~", "…"]
    for p in punctuation:
        if p not in vocab:
            vocab.append(p)

    return vocab

def convert_text_to_phonemes(text: str) -> list[str]:
    """
    Convert text to standard phonemes (Jamos) using g2pk + decomposition.
    """
    # 1. Text to Pronunciation (Hangul Syllables) via g2pk
    pronunciation = _text_to_pronunciation(text)
    
    # 2. Decompose into Jamos
    phonemes = []
    # Split by space to handle word boundaries if needed, but here we just want the sequence
    # _text_to_pronunciation preserves spaces
    
    for char in pronunciation:
        if char == ' ':
            phonemes.append(' ')
        else:
            # Decompose syllable
            jamos = _syllable_to_phonemes(char)
            phonemes.extend(jamos)
            
    return phonemes

def main():
    # Configuration
    dataset_name = "KSS"
    tokenizer_type = "kor_phoneme"
    
    # Paths
    project_root = Path(os.getcwd())
    data_root = project_root / "data"
    dataset_dir = data_root / "KSS"
    transcript_path = dataset_dir / "transcript.v.1.4.txt"
    
    # Output directory: data/KSS_phoneme
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
            phoneme_tokens = convert_text_to_phonemes(text_content)
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

    # Generate and save vocab.txt
    print("\nGenerating vocab.txt ...")
    vocab = generate_phoneme_vocab()
    
    # Ensure all used tokens are in vocab (except maybe rare symbols which will be mapped to space/0)
    # Let's verify commonly used punctuation in KSS
    
    with open(save_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for v in vocab:
            f.write(v + "\n")

    print(f"Vocab size: {len(vocab)}")
    print(f"Total duration: {sum(duration_list) / 3600:.2f} hours")
    print("Done!")

if __name__ == "__main__":
    main()
