import argparse
import math
from collections import Counter
from pathlib import Path
import os
import sys

import numpy as np
from datasets import Dataset as Dataset_
from tqdm import tqdm

def calculate_gini(counts):
    """
    지니 계수 계산 (0: 완전 평등, 1: 완전 불평등)
    """
    if not counts:
        return 0.0
    
    # 빈도수 배열 (오름차순 정렬)
    array = np.sort(np.array(list(counts.values())))
    n = len(array)
    
    # Gini coefficient calculation using the sorted values
    index = np.arange(1, n + 1)
    return ((2 * index - n - 1) * array).sum() / (n * array.sum())

def calculate_renyi_entropy(counts, alpha=2.5):
    """
    Renyi Entropy 계산 (alpha=2.5: Based on existing analysis convention)
    """
    total_count = sum(counts.values())
    if total_count == 0:
        return 0.0
    
    probs = np.array([c / total_count for c in counts.values()])
    
    if alpha == 1.0:
        # Shannon Entropy limit
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    sum_probs_alpha = np.sum(probs ** alpha)
    return (1 / (1 - alpha)) * np.log2(sum_probs_alpha)

def calculate_shannon_entropy(counts):
    """Shannon Entropy (Renyi alpha -> 1)"""
    return calculate_renyi_entropy(counts, alpha=1.0)

def get_ngrams(tokens, n):
    """토큰 리스트에서 n-gram 생성 (문장 경계 넘지 않음)"""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def analyze_dataset(dataset_name, data_root=None, silent=False):
    if data_root is None:
        data_root = Path(os.getcwd()) / "data"
    
    dataset_path = data_root / dataset_name
    arrow_path = dataset_path / "raw.arrow"
    
    if not arrow_path.exists():
        print(f"Error: {arrow_path} not found.")
        return

    print(f"Loading dataset: {dataset_name}")
    try:
        ds = Dataset_.from_file(str(arrow_path))
    except Exception as e:
        print(f"Failed to load arrow: {e}")
        return

    # Counters
    ngram_counters = {
        1: Counter(),
        2: Counter(),
        3: Counter()
    }

    # Filter characters: ignore space, punctuation, and known special tokens if needed
    # Usually we want to analyze distribution of meaningful phonemes/graphemes.
    ignored_tokens = {' ', '.', ',', '?', '!', '~', '…', 'waiting...', 'unintelligible'}

    # Process
    print("Counting N-grams (ignoring spaces/punctuations)...")
    for row in tqdm(ds):
        text = row["text"] # List of tokens
        
        # Filter tokens
        filtered_text = [t for t in text if t not in ignored_tokens and t.strip()]
        
        if not filtered_text:
            continue

        # 1-gram
        ngram_counters[1].update(filtered_text)
        
        # 2-gram
        bigrams = get_ngrams(filtered_text, 2)
        ngram_counters[2].update(bigrams)
        
        # 3-gram
        trigrams = get_ngrams(filtered_text, 3)
        ngram_counters[3].update(trigrams)

    # Calculate Metrics
    if not silent:
        print("\n" + "="*80)
        print(f"Analysis Result for: {dataset_name}")
        print("="*80)
        
        header = f"{'N-gram':<10} | {'Unique(Vocab)':<12} | {'Total Count':<12} | {'Gini':<8} | {'Shannon':<8} | {'Renyi(a=2.5)':<12} | {'Eff(Renyi)':<8}"
        print(header)
        print("-" * len(header))

    stats_result = {}

    for n in [1, 2, 3]:
        counter = ngram_counters[n]
        vocab_size = len(counter)
        total_count = sum(counter.values())
        
        gini = calculate_gini(counter)
        shannon = calculate_shannon_entropy(counter)
        renyi = calculate_renyi_entropy(counter, alpha=2.5) # Using alpha=2.5 to match summarize_dataset_stats.py
        
        # Efficiency = Entropy / Max_Entropy
        # Max Entropy = log2(Vocab Size)
        max_entropy = math.log2(vocab_size) if vocab_size > 0 else 1
        efficiency = renyi / max_entropy if max_entropy > 0 else 0
        
        stats_result[n] = {
            "vocab": vocab_size,
            "count": total_count,
            "gini": gini,
            "shannon": shannon,
            "renyi": renyi,
            "eff": efficiency
        }

        if not silent:
            print(f"{n}-gram     | {vocab_size:<12} | {total_count:<12} | {gini:.4f}   | {shannon:.4f}   | {renyi:.4f}     | {efficiency:.4f}")

    if not silent:
        print("="*80 + "\n")
        
        # Top 10 frequencies per N-gram (Optional verification)
        for n in [1, 2, 3]:
            print(f"Top 5 {n}-grams:")
            for item, count in ngram_counters[n].most_common(5):
                # Tuple to string for display
                if n == 1:
                    display = item
                else:
                    display = str(item)
                print(f"  {display}: {count}")
            print("-" * 20)
            
    return stats_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset folder in data/")
    args = parser.parse_args()
    
    analyze_dataset(args.dataset_name)
