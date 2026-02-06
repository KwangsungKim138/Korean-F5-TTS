
import os
from f5_tts.model.utils import (
    PHONEME_CONSONANTS,
    PHONEME_VOWELS,
    PHONEMES_I,
    PHONEMES_P,
    PHONEMES_C,
    MARK_INIT,
    MARK_PAL,
    MARK_CODA
)

def generate_vocab():
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

def save_vocab(vocab, path="vocab.txt"):
    with open(path, "w", encoding="utf-8") as f:
        for v in vocab:
            f.write(v + "\n")
    print(f"Saved {len(vocab)} tokens to {path}")

if __name__ == "__main__":
    vocab = generate_vocab()
    save_vocab(vocab, "data/korean_allophone_vocab.txt")
