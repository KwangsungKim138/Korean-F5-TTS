import os
import sys
import shutil
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
    MARK_CODA
)

def generate_korean_vocab():
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
            
    # 2. Allophones
    for voiceless in PHONEMES_I: # 어두 초성
        token = voiceless + MARK_INIT
        if token not in vocab:
            vocab.append(token)
            
    for palatal in PHONEMES_P: # 구개음화
        token = palatal + MARK_PAL
        if token not in vocab:
            vocab.append(token)
            
    for coda in PHONEMES_C: # 종성
        token = coda + MARK_CODA
        if token not in vocab:
            vocab.append(token)
    return vocab

def main():
    # Configuration
    dataset_name = "KSS"
    tokenizer_type = "kor_allophone"
    
    # Paths
    # Assuming running from project root
    project_root = Path(os.getcwd())
    data_root = project_root / "data"
    dataset_dir = data_root / "KSS"
    transcript_path = dataset_dir / "transcript.v.1.4.txt"
    
    # Output directory: data/KSS_kor_allophone
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
            allophone_tokens = allophone_tokens_list[0] # Get the first (and only) result
            
            # Update vocab set
            vocab_set.update(allophone_tokens)
            
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

    # Save vocab.txt
    print("\nGenerating vocab.txt ...")
    
    with open(save_dir / "vocab.txt", "w", encoding="utf-8") as f:
        # Ensure space is at index 0
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
