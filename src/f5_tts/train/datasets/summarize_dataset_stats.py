"""
Summarize statistics for multiple datasets in a single table.
Outputs Gini coefficient, Rényi Entropy, and Efficiency for both All symbols and Consonants only.
"""
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset

sys.path.insert(0, os.getcwd())

# --- Configuration ---
DATASETS_TO_ANALYZE = [
    "data/KSS_n2gk_grapheme",
    "data/KSS_n2gk_phoneme",
    "data/KSS_n2gk_allophone",
    "data/KSS_n2gk_i_only",
    "data/KSS_n2gk_c_only",
    "data/KSS_n2gk_n_only",
    "data/KSS_n2gk_i_and_c",
    "data/KSS_n2gk_i_and_n",
    "data/KSS_n2gk_inf",
    "data/KSS_n2gk_nf",
    "data/KSS_n2gk_efficient_allophone",
    "data/KSS_1h_n2gk_grapheme",
    "data/KSS_1h_n2gk_phoneme",
    "data/KSS_1h_n2gk_allophone",
    "data/KSS_1h_n2gk_inf",
]

# --- Constants & Regex ---
VALID_TOKEN_PATTERN = re.compile(r'^[가-힣ㄱ-ㅎㅏ-ㅣ][ⁱᶜʲ]?$')
VALID_CONSONANT_PATTERN = re.compile(r'^[ㄱ-ㅎ][ⁱᶜʲ]?$')


# --- Calculation Functions ---
def calculate_gini(frequencies):
    if not frequencies:
        return 0.0
    counts = np.array(frequencies, dtype=np.float64)
    counts = np.sort(counts)
    n = len(counts)
    if n == 0 or counts.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    numerator = ((2 * index - n - 1) * counts).sum()
    denominator = n * counts.sum()
    return numerator / denominator


def calculate_renyi_entropy(frequencies: list, alpha: float = 2.5) -> float:
    if not frequencies:
        return 0.0
    counts = np.array(frequencies, dtype=np.float64)
    total_count = counts.sum()
    if total_count == 0:
        return 0.0
    probs = counts / total_count
    sum_probs_alpha = np.sum(probs ** alpha)
    if alpha == 1.0:
        probs_nonzero = probs[probs > 0]
        entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))
    else:
        entropy = (1 / (1 - alpha)) * np.log2(sum_probs_alpha)
    return entropy


def calculate_renyi_efficiency(entropy: float, vocab_size: int) -> float:
    if vocab_size <= 1:
        return 0.0
    max_entropy = math.log2(vocab_size)
    return entropy / max_entropy


def analyze_dataset(dataset_path: str):
    path = Path(dataset_path) / "raw.arrow"
    if not path.exists():
        return None

    try:
        ds = Dataset.from_file(str(path))
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

    if "text" not in ds.column_names:
        return None

    counter_all = Counter()
    counter_cons = Counter()

    for row in ds:
        text = row["text"]
        tokens = []
        if isinstance(text, list):
            tokens = text
        elif isinstance(text, str):
            tokens = list(text)

        valid_tokens = [t for t in tokens if VALID_TOKEN_PATTERN.match(t)]
        valid_consonants = [t for t in tokens if VALID_CONSONANT_PATTERN.match(t)]
        
        counter_all.update(valid_tokens)
        counter_cons.update(valid_consonants)

    # All Symbols Stats
    counts_all = list(counter_all.values())
    vocab_size_all = len(counts_all)
    gini_all = calculate_gini(counts_all)
    renyi_all = calculate_renyi_entropy(counts_all, alpha=2.5)
    eff_all = calculate_renyi_efficiency(renyi_all, vocab_size_all)

    # Consonants Stats
    counts_cons = list(counter_cons.values())
    vocab_size_cons = len(counts_cons)
    gini_cons = calculate_gini(counts_cons)
    renyi_cons = calculate_renyi_entropy(counts_cons, alpha=2.5)
    eff_cons = calculate_renyi_efficiency(renyi_cons, vocab_size_cons)

    return {
        "name": Path(dataset_path).name.replace("KSS_n2gk_", ""),
        "vocab_all": vocab_size_all,
        "gini_all": gini_all,
        "renyi_all": renyi_all,
        "eff_all": eff_all,
        "vocab_cons": vocab_size_cons,
        "gini_cons": gini_cons,
        "renyi_cons": renyi_cons,
        "eff_cons": eff_cons,
    }


def main():
    results = []
    print("Analyzing datasets...")
    
    for ds_path in DATASETS_TO_ANALYZE:
        print(f"Processing {ds_path}...", end="\r")
        stats = analyze_dataset(ds_path)
        if stats:
            results.append(stats)
        else:
            # Add placeholder for missing dataset
            results.append({
                "name": Path(ds_path).name.replace("KSS_n2gk_", "") + " (Not Found)",
                "vocab_all": 0, "gini_all": 0.0, "renyi_all": 0.0, "eff_all": 0.0,
                "vocab_cons": 0, "gini_cons": 0.0, "renyi_cons": 0.0, "eff_cons": 0.0,
            })
    print(" " * 50)  # Clear line

    # --- Print Table ---
    # Header
    header = f"{'Dataset':<15} | {'Vocab':<6} {'Gini':<6} {'Renyi':<6} {'Eff':<6} | {'C.Voc':<6} {'C.Gini':<6} {'C.Renyi':<7} {'C.Eff':<6}"
    sep = "-" * len(header)
    
    print("\n" + sep)
    print(header)
    print(sep)

    for r in results:
        line = (
            f"{r['name']:<15} | "
            f"{r['vocab_all']:<6} {r['gini_all']:.3f}  {r['renyi_all']:.3f}  {r['eff_all']:.3f}  | "
            f"{r['vocab_cons']:<6} {r['gini_cons']:.3f}   {r['renyi_cons']:.3f}   {r['eff_cons']:.3f}"
        )
        print(line)
    print(sep + "\n")
    print("* Renyi Entropy (alpha=2.5), Efficiency = Renyi / log2(Vocab)")


if __name__ == "__main__":
    main()
