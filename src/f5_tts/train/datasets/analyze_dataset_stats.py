"""
Analyze symbol statistics of a prepared dataset (raw.arrow).
Calculates frequency counts and Gini coefficient.
Output: stats.txt in the dataset directory.
"""
import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

import math
import numpy as np
from datasets import Dataset

sys.path.insert(0, os.getcwd())


# Regex to match Korean char (Hangul Syllables or Jamo) optionally followed by a tag (init/coda/pal)
# Range: 가-힣 (AC00-D7A3), ㄱ-ㅎㅏ-ㅣ (3131-3163 approx)
# Tags: ⁱ, ᶜ, ʲ
VALID_TOKEN_PATTERN = re.compile(r'^[가-힣ㄱ-ㅎㅏ-ㅣ][ⁱᶜʲ]?$')
VALID_CONSONANT_PATTERN = re.compile(r'^[ㄱ-ㅎ][ⁱᶜʲ]?$')

def calculate_gini(frequencies):
    """
    Calculate Gini coefficient of a frequency distribution.
    frequencies: list of counts (integers)
    """
    if not frequencies:
        return 0.0
    
    counts = np.array(frequencies, dtype=np.float64)
    # Sort ascending for the formula
    counts = np.sort(counts)
    n = len(counts)
    
    if n == 0 or counts.sum() == 0:
        return 0.0

    # Gini coefficient formula:
    # G = sum((2i - n - 1) * xi) / (n * sum(xi))
    # where xi is the frequency sorted in ascending order
    
    index = np.arange(1, n + 1)
    numerator = ((2 * index - n - 1) * counts).sum()
    denominator = n * counts.sum()
    
    return numerator / denominator


def calculate_renyi_entropy(frequencies: list, alpha: float = 2.5) -> float:
    """
    Rényi Entropy calculation based on Zouhar et al. (2023).
    H_alpha(p) = 1/(1-alpha) * log2(sum(p_i^alpha))
    """
    if not frequencies:
        return 0.0
        
    counts = np.array(frequencies, dtype=np.float64)
    total_count = counts.sum()
    
    if total_count == 0:
        return 0.0
        
    probs = counts / total_count
    
    # sum(p_i ^ alpha)
    sum_probs_alpha = np.sum(probs ** alpha)
    
    if alpha == 1.0:
        # Shannon Entropy limit case
        # Avoid log(0)
        probs_nonzero = probs[probs > 0]
        entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))
    else:
        entropy = (1 / (1 - alpha)) * np.log2(sum_probs_alpha)
        
    return entropy


def calculate_renyi_efficiency(entropy: float, vocab_size: int) -> float:
    """
    Rényi Efficiency = Entropy / Max Possible Entropy (log2(|V|))
    """
    if vocab_size <= 1:
        return 0.0
        
    max_entropy = math.log2(vocab_size)
    return entropy / max_entropy


def main():
    parser = argparse.ArgumentParser(description="Analyze symbol statistics of a prepared dataset.")
    parser.add_argument("dataset_dir", type=Path, help="Path to the dataset directory containing raw.arrow (e.g., data/KSS_n2gk_i_only)")
    parser.add_argument("--include-space", action="store_true", help="Include space character ' ' in statistics (default: False)")
    args = parser.parse_args()

    dataset_path = args.dataset_dir
    arrow_path = dataset_path / "raw.arrow"
    
    if not arrow_path.exists():
        print(f"Error: {arrow_path} does not exist.")
        return

    print(f"Loading dataset from {arrow_path}...")
    try:
        # Load directly from arrow file
        ds = Dataset.from_file(str(arrow_path))
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Count symbols
    counter1 = Counter()
    counter2 = Counter()
    print("Counting symbols...")
    
    # Check dataset structure
    if "text" not in ds.column_names:
        print("Error: Dataset does not contain a 'text' column.")
        return

    for row in ds:
        text = row["text"]
        tokens = []
        if isinstance(text, list):
            # If tokenized list (e.g. allophone tokens)
            tokens = text
        elif isinstance(text, str):
            # If raw string (e.g. grapheme string)
            tokens = list(text)

        # Filter tokens: only valid Korean chars (+tags)
        # Note: space is automatically excluded by the regex pattern
        valid_tokens = [t for t in tokens if VALID_TOKEN_PATTERN.match(t)]
        valid_consonants = [t for t in tokens if VALID_CONSONANT_PATTERN.match(t)]
        counter1.update(valid_tokens)
        counter2.update(valid_consonants)

    # Sort by count descending
    sorted_stats1 = counter1.most_common()
    sorted_stats2 = counter2.most_common()
    
    # Calculate Gini
    counts1 = [count for _, count in sorted_stats1]
    gini1 = calculate_gini(counts1)
    renyi1 = calculate_renyi_entropy(counts1, alpha=2.5)
    eff1 = calculate_renyi_efficiency(renyi1, len(counts1))
    
    # Calculate Gini (consonant only)
    counts2 = [count for _, count in sorted_stats2]
    gini2 = calculate_gini(counts2)
    renyi2 = calculate_renyi_entropy(counts2, alpha=2.5)
    eff2 = calculate_renyi_efficiency(renyi2, len(counts2))
    
    print(f"Total unique symbols: {len(sorted_stats1)}")
    print(f"Total unique consonants: {len(sorted_stats2)}")
    print(f"Total tokens: {sum(counts1)}")
    print(f"Total consonant tokens: {sum(counts2)}")
    print(f"Gini coefficient: {gini1:.6f}")
    print(f"Rényi Entropy (α=2.5): {renyi1:.4f}")
    print(f"Rényi Efficiency: {eff1:.4f}")
    print(f"Gini coefficient(consonant): {gini2:.6f}")
    print(f"Rényi Entropy(consonant) (α=2.5): {renyi2:.4f}")
    print(f"Rényi Efficiency(consonant): {eff2:.4f}")
    
    # Output to file
    output_path = dataset_path / "stats.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== All Symbols (Hangul + Tags) ===\n")
        f.write(f"Gini coefficient: {gini1:.6f}\n")
        f.write(f"Rényi Entropy (α=2.5): {renyi1:.4f} bits\n")
        f.write(f"Rényi Efficiency: {eff1:.4f}\n")
        f.write(f"Unique symbols: {len(sorted_stats1)}\n")
        f.write(f"Total tokens: {sum(counts1)}\n")
        f.write("-" * 30 + "\n")
        for char, count in sorted_stats1:
            f.write(f"{char}\t{count}\n")
        
        f.write("\n\n=== Consonants Only (ㄱ-ㅎ + Tags) ===\n")
        f.write(f"Gini coefficient: {gini2:.6f}\n")
        f.write(f"Rényi Entropy (α=2.5): {renyi2:.4f} bits\n")
        f.write(f"Rényi Efficiency: {eff2:.4f}\n")
        f.write(f"Unique symbols: {len(sorted_stats2)}\n")
        f.write(f"Total tokens: {sum(counts2)}\n")
        f.write("-" * 30 + "\n")
        for char, count in sorted_stats2:
            f.write(f"{char}\t{count}\n")
            
    print(f"Stats saved to {output_path}")


if __name__ == "__main__":
    main()
