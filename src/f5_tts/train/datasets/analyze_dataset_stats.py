"""
Analyze symbol statistics of a prepared dataset (raw.arrow).
Calculates frequency counts and Gini coefficient.
Output: stats.txt in the dataset directory.
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset

sys.path.insert(0, os.getcwd())


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


def main():
    parser = argparse.ArgumentParser(description="Analyze symbol statistics of a prepared dataset.")
    parser.add_argument("dataset_dir", type=Path, help="Path to the dataset directory containing raw.arrow (e.g., data/KSS_n2gk_i_only)")
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
    counter = Counter()
    print("Counting symbols...")
    
    # Check dataset structure
    if "text" not in ds.column_names:
        print("Error: Dataset does not contain a 'text' column.")
        return

    for row in ds:
        text = row["text"]
        if isinstance(text, list):
            # If tokenized list (e.g. allophone tokens)
            counter.update(text)
        elif isinstance(text, str):
            # If raw string (e.g. grapheme string)
            counter.update(list(text))
        else:
            pass

    # Sort by count descending
    sorted_stats = counter.most_common()
    
    # Calculate Gini
    counts = [count for _, count in sorted_stats]
    gini = calculate_gini(counts)
    
    print(f"Total unique symbols: {len(sorted_stats)}")
    print(f"Total tokens: {sum(counts)}")
    print(f"Gini coefficient: {gini:.6f}")
    
    # Output to file
    output_path = dataset_path / "stats.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"gini coefficient: {gini:.6f}\n")
        for char, count in sorted_stats:
            f.write(f"{char}\t{count}\n")
            
    print(f"Stats saved to {output_path}")


if __name__ == "__main__":
    main()
