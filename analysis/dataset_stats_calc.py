"""
Data statistics for the paper: token proportions and normalized Shannon entropy.

Reads train_full.txt and test.txt from data/KSS/, applies grapheme/phoneme/allophone
conversions (using project utils). All metrics use content_tokens only (empty string,
space, and punctuation . , ! ? ~ … excluded). Computes:
  1. Content token proportion (vs raw token count).
  2. Allophone only: proportion of content tokens with variation marks (ⁱ, ᶜ, ʲ).
  3. Normalized Shannon entropy (unique types, entropy) per conversion type.
  4. Consonant (자음) only: (1) consonant / all tokens, (2) consonant / content tokens,
     (3) consonant token distribution; outputs *_consonant_counts.txt.

Run from project root: PYTHONPATH=src python analysis/allophone_stats.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Resolve project root and ensure src is on path for standalone run
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from collections import Counter

from f5_tts.model.utils import (
    MARK_CODA,
    MARK_INIT,
    MARK_PAL,
    PHONEME_CONSONANTS,
    PHONEME_VOWELS,
    PHONEMES_C,
    PHONEMES_I,
    PHONEMES_P,
    GRAPHEME_CHOSEONG,
    GRAPHEME_JUNGSEONG,
    GRAPHEME_JONGSEONG,
    _syllable_to_phonemes,
    _text_to_pronunciation,
    convert_char_to_allophone,
    convert_char_to_grapheme,
)

# 1. Grapheme pools (utils: choseong + jungseong + jongseong; consonant = choseong + jongseong)
_graf_full = set(GRAPHEME_CHOSEONG) | set(GRAPHEME_JUNGSEONG) | {x for x in GRAPHEME_JONGSEONG if x}
_graf_consonant = set(GRAPHEME_CHOSEONG) | {x for x in GRAPHEME_JONGSEONG if x}
POOL_GRAPHEME_FULL = _graf_full
POOL_GRAPHEME_CONSONANT = _graf_consonant

# 2. Phoneme pools
POOL_PHONEME_CONSONANT = set(PHONEME_CONSONANTS)
POOL_PHONEME_VOWEL = set(PHONEME_VOWELS)
POOL_PHONEME_FULL = POOL_PHONEME_CONSONANT | POOL_PHONEME_VOWEL

# 3. Allophone pools (vowel = phoneme vowels; consonant = phoneme consonants + I+INIT, P+PAL, C+CODA)
POOL_ALLOPHONE_VOWEL = set(PHONEME_VOWELS)
POOL_ALLOPHONE_CONSONANT = (
    set(PHONEME_CONSONANTS)
    | {p + MARK_INIT for p in PHONEMES_I}
    | {p + MARK_PAL for p in PHONEMES_P}
    | {p + MARK_CODA for p in PHONEMES_C}
)
POOL_ALLOPHONE_FULL = POOL_ALLOPHONE_VOWEL | POOL_ALLOPHONE_CONSONANT

# Punctuation and non-content tokens to filter out before stats (space, punct = delimiters, not segment counts)
PUNCT = {".", ",", "!", "?", "~", "…"}


def filter_content(tokens: list[str]) -> list[str]:
    """Remove empty string, space, and punctuation. All stats use this list (content/segment tokens only)."""
    return [t for t in tokens if t != "" and t != " " and t not in PUNCT]


def text_to_phoneme_tokens(text: str) -> list[str]:
    """Convert one line to phoneme (jamo) token list via G2P + decomposition."""
    pronunciation = _text_to_pronunciation(text)
    phonemes = []
    for char in pronunciation:
        if char == " ":
            phonemes.append(" ")
        else:
            for p in _syllable_to_phonemes(char):
                if p:  # skip empty string (e.g. no coda)
                    phonemes.append(p)
    return phonemes


def load_texts(path: Path) -> list[str]:
    """Load text field (index 2) from KSS metadata: path|orig|text|..."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) > 2:
                texts.append(parts[2])
    return texts


def content_proportion(tokens: list[str]) -> float:
    """Proportion of tokens that are content (not '', space, or punctuation)."""
    if not tokens:
        return 0.0
    n_content = len(filter_content(tokens))
    return n_content / len(tokens)


def is_consonant_token(tok: str, consonant_pool: set[str]) -> bool:
    """True if token is in the given consonant pool (type-specific: grapheme / phoneme / allophone)."""
    return tok in consonant_pool


def allophone_marked_ratio(content_tokens: list[str]) -> float:
    """Proportion of content tokens that contain any allophone mark (ⁱ, ᶜ, ʲ). Denominator = len(content_tokens)."""
    if not content_tokens:
        return 0.0
    marks = (MARK_INIT, MARK_CODA, MARK_PAL)
    marked = sum(1 for t in content_tokens if any(m in t for m in marks))
    return marked / len(content_tokens)


def evenness_stats(counter: Counter) -> dict:
    """Unique types, total count, normalized Shannon entropy (0--1), and min/max counts."""
    total = sum(counter.values())
    n_types = len(counter)
    if total == 0 or n_types == 0:
        return {"n_types": 0, "total": 0, "entropy_norm": 0.0, "min_count": 0, "max_count": 0}

    entropy = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    max_entropy = math.log2(n_types) if n_types else 0
    entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0

    counts = list(counter.values())
    return {
        "n_types": n_types,
        "total": total,
        "entropy_norm": round(entropy_norm, 4),
        "min_count": min(counts),
        "max_count": max(counts),
    }


def run_one_split(
    name: str,
    texts: list[str],
    out_dir: Path | None,
) -> dict:
    """Compute and print stats; return dict of numeric results for CSV."""
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    n_texts = len(texts)

    print("  Converting to grapheme...")
    grapheme_seqs = [convert_char_to_grapheme([t])[0] for t in texts]
    print("  Converting to phoneme (G2P, may take a while)...")
    phoneme_seqs = []
    for i, t in enumerate(texts):
        phoneme_seqs.append(text_to_phoneme_tokens(t))
        if (i + 1) % 2000 == 0 or (i + 1) == n_texts:
            print(f"    phoneme {i + 1}/{n_texts}")
    print("  Converting to allophone...")
    allophone_seqs = []
    for i, t in enumerate(texts):
        allophone_seqs.append(convert_char_to_allophone([t])[0])
        if (i + 1) % 2000 == 0 or (i + 1) == n_texts:
            print(f"    allophone {i + 1}/{n_texts}")

    def flatten(seqs: list[list[str]]) -> list[str]:
        return [t for seq in seqs for t in seq]

    grapheme_tokens = flatten(grapheme_seqs)
    phoneme_tokens = flatten(phoneme_seqs)
    allophone_tokens = flatten(allophone_seqs)

    # Filtering: content tokens only (remove '', space, punctuation) — same criterion for all three
    grapheme_content = filter_content(grapheme_tokens)
    phoneme_content = filter_content(phoneme_tokens)
    allophone_content = filter_content(allophone_tokens)

    # Metric 1: proportion of raw tokens that are content (for reference)
    print("\n1. Content token proportion (exclude '', space, . , ! ? ~ …):")
    pr_g = content_proportion(grapheme_tokens)
    pr_p = content_proportion(phoneme_tokens)
    pr_a = content_proportion(allophone_tokens)
    print(f"   Grapheme:  {pr_g:.2%}  (n_content={len(grapheme_content)}, n_total={len(grapheme_tokens)})")
    print(f"   Phoneme:   {pr_p:.2%}  (n_content={len(phoneme_content)}, n_total={len(phoneme_tokens)})")
    print(f"   Allophone: {pr_a:.2%}  (n_content={len(allophone_content)}, n_total={len(allophone_tokens)})")

    # Metric 2: marker ratio — denominator = content_tokens count (allophone only)
    print("\n2. Allophone: proportion of content tokens with variation marks (ⁱ, ᶜ, ʲ) [denom = content_tokens]:")
    amr = allophone_marked_ratio(allophone_content)
    print(f"   {amr:.2%}")

    # Metric 3: normalized Shannon entropy — full pool as alphabet; pool types not seen in data count as 0
    FULL_POOLS = {
        "Grapheme": POOL_GRAPHEME_FULL,
        "Phoneme": POOL_PHONEME_FULL,
        "Allophone": POOL_ALLOPHONE_FULL,
    }
    print("\n3. Token distribution — normalized Shannon entropy (full pool as alphabet; unseen pool types = 0):")
    results = {"split": name, "pure_grapheme": pr_g, "pure_phoneme": pr_p, "pure_allophone": pr_a, "allophone_marked": amr}
    for label, content in [
        ("Grapheme", grapheme_content),
        ("Phoneme", phoneme_content),
        ("Allophone", allophone_content),
    ]:
        pool = FULL_POOLS[label]
        content_in_pool = [t for t in content if t in pool]
        c = Counter(content_in_pool)
        # Include all pool types so normalized Shannon entropy is over full alphabet (unseen = 0)
        c_full = Counter({k: c.get(k, 0) for k in pool})
        s = evenness_stats(c_full)
        key = label.lower()
        results[f"{key}_types"] = s["n_types"]  # full pool size
        results[f"{key}_total"] = s["total"]
        results[f"{key}_entropy_norm"] = s["entropy_norm"]
        results[f"{key}_min_count"] = s["min_count"]
        results[f"{key}_max_count"] = s["max_count"]
        n_out = len(content) - len(content_in_pool)
        out_str = f"  (n_out_of_pool={n_out})" if n_out else ""
        print(f"   {label}: unique={s['n_types']}, total={s['total']}, entropy_norm={s['entropy_norm']:.4f}, min_count={s['min_count']}, max_count={s['max_count']}{out_str}")

    # Consonant-only stats (자음) — pool per type: grapheme (choseong+jongseong), phoneme (PHONEME_CONSONANTS), allophone (+ variation marks)
    CONSONANT_POOLS = {
        "Grapheme": POOL_GRAPHEME_CONSONANT,
        "Phoneme": POOL_PHONEME_CONSONANT,
        "Allophone": POOL_ALLOPHONE_CONSONANT,
    }
    print("\n4. Consonant tokens (자음) — pool-based (utils.py):")
    for label, full_tok, content in [
        ("Grapheme", grapheme_tokens, grapheme_content),
        ("Phoneme", phoneme_tokens, phoneme_content),
        ("Allophone", allophone_tokens, allophone_content),
    ]:
        pool = CONSONANT_POOLS[label]
        n_full = len(full_tok)
        n_content = len(content)
        cons_full = [t for t in full_tok if is_consonant_token(t, pool)]
        cons_content = [t for t in content if is_consonant_token(t, pool)]
        n_cons_full = len(cons_full)
        n_cons_content = len(cons_content)
        ratio1 = n_cons_full / n_full if n_full else 0.0
        ratio2 = n_cons_content / n_content if n_content else 0.0
        key = label.lower()
        results[f"{key}_consonant_ratio_all"] = ratio1
        results[f"{key}_consonant_ratio_content"] = ratio2
        results[f"{key}_consonant_types"] = len(set(cons_content))
        results[f"{key}_consonant_total"] = n_cons_content
        print(f"   {label}:")
        print(f"      (1) Consonant / all tokens:     {ratio1:.2%}  (n_consonant={n_cons_full}, n_all={n_full})")
        print(f"      (2) Consonant / content tokens: {ratio2:.2%}  (n_consonant={n_cons_content}, n_content={n_content})")
        c_cons = Counter(cons_content)
        # Normalized Shannon entropy over full consonant pool (unseen = 0)
        c_cons_full = Counter({k: c_cons.get(k, 0) for k in pool})
        s_cons = evenness_stats(c_cons_full)
        print(f"      (3) Consonant distribution: pool_size={s_cons['n_types']}, total={s_cons['total']}, entropy_norm={s_cons['entropy_norm']:.4f}")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        print("  Writing count files (full pool only)...")
        _full_pools = {"grapheme": POOL_GRAPHEME_FULL, "phoneme": POOL_PHONEME_FULL, "allophone": POOL_ALLOPHONE_FULL}
        for label, content in [
            ("grapheme", grapheme_content),
            ("phoneme", phoneme_content),
            ("allophone", allophone_content),
        ]:
            content_in_pool = [t for t in content if t in _full_pools[label]]
            c = Counter(content_in_pool)
            out_file = out_dir / f"{name.replace(' ', '_')}_{label}_counts.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                for tok, cnt in c.most_common():
                    f.write(f"{cnt}\t{tok}\n")
        # Consonant-only count files (use same pool per type)
        _pools = {"grapheme": POOL_GRAPHEME_CONSONANT, "phoneme": POOL_PHONEME_CONSONANT, "allophone": POOL_ALLOPHONE_CONSONANT}
        for label, content in [
            ("grapheme", grapheme_content),
            ("phoneme", phoneme_content),
            ("allophone", allophone_content),
        ]:
            cons_content = [t for t in content if is_consonant_token(t, _pools[label])]
            c_cons = Counter(cons_content)
            out_file = out_dir / f"{name.replace(' ', '_')}_{label}_consonant_counts.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                for tok, cnt in c_cons.most_common():
                    f.write(f"{cnt}\t{tok}\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="KSS data statistics for paper (grapheme/phoneme/allophone).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing train_full.txt and test.txt (default: project_root/data/KSS)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write token-count files and plots (default: analysis/output)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or (PROJECT_ROOT / "data" / "KSS")
    out_dir = args.output_dir or (SCRIPT_DIR / "output")

    train_path = data_dir / "train_full.txt"
    test_path = data_dir / "test.txt"
    print(f"Data dir: {data_dir}")
    if not train_path.is_file():
        print(f"Error: {train_path} not found.")
        sys.exit(1)
    if not test_path.is_file():
        print(f"Error: {test_path} not found.")
        sys.exit(1)

    print("Loading metadata...")
    train_texts = load_texts(train_path)
    test_texts = load_texts(test_path)
    print(f"Loaded {len(train_texts)} lines from train_full.txt, {len(test_texts)} from test.txt.")

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n--- train_full ---")
    rows = [run_one_split("train_full", train_texts, out_dir)]
    print("\n--- test ---")
    rows.append(run_one_split("test", test_texts, out_dir))

    # Save numeric stats for paper table (CSV)
    print("\nWriting CSV...")
    import csv
    if rows:
        keys = list(rows[0].keys())
        csv_path = out_dir / "data_stats.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved numeric stats: {csv_path}")

    # Visualization: rank vs count (allophone) and normalized Shannon entropy comparison
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping plots)")
        return

    print("Plotting allophone token distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (split_name, texts) in zip(axes, [("train_full", train_texts), ("test", test_texts)]):
        allophone_seqs = [convert_char_to_allophone([t])[0] for t in texts]
        tokens = filter_content([t for seq in allophone_seqs for t in seq])
        c = Counter(tokens)
        counts = [cnt for _, cnt in c.most_common()]
        ax.plot(range(len(counts)), counts, alpha=0.8)
        ax.set_xlabel("Token rank")
        ax.set_ylabel("Count")
        ax.set_title(f"Allophone token distribution ({split_name}, content only)")
        ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(out_dir / "allophone_token_distribution.png", dpi=150)
    plt.close()
    print(f"Saved plot: {out_dir / 'allophone_token_distribution.png'}")

    # 1) Normalized Shannon entropy — full jamo pool (unseen = 0)
    print("Plotting normalized Shannon entropy (full pool)...")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    full_pools = [POOL_GRAPHEME_FULL, POOL_PHONEME_FULL, POOL_ALLOPHONE_FULL]
    grapheme_t = [t for seq in [convert_char_to_grapheme([t])[0] for t in train_texts] for t in seq]
    grapheme_t = [t for t in filter_content(grapheme_t) if t in POOL_GRAPHEME_FULL]
    phoneme_t = [t for seq in [text_to_phoneme_tokens(t) for t in train_texts] for t in seq]
    phoneme_t = [t for t in filter_content(phoneme_t) if t in POOL_PHONEME_FULL]
    allophone_t = [t for seq in [convert_char_to_allophone([t])[0] for t in train_texts] for t in seq]
    allophone_t = [t for t in filter_content(allophone_t) if t in POOL_ALLOPHONE_FULL]
    contents = [grapheme_t, phoneme_t, allophone_t]
    entropies_full = []
    for content, pool in zip(contents, full_pools):
        c = Counter(content)
        c_full = Counter({k: c.get(k, 0) for k in pool})
        entropies_full.append(evenness_stats(c_full)["entropy_norm"])
    labels = ["Grapheme", "Phoneme", "Allophone"]
    ax1.bar(labels, entropies_full, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax1.set_ylabel("Normalized Shannon entropy (0–1)")
    ax1.set_title("Normalized Shannon entropy — full jamo pool (train_full, unseen=0)")
    ax1.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_dir / "evenness_full_pool.png", dpi=150)
    plt.close()
    print(f"Saved plot: {out_dir / 'evenness_full_pool.png'}")

    # 2) Normalized Shannon entropy — consonant pool only (unseen = 0)
    print("Plotting normalized Shannon entropy (consonant pool only)...")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    cons_pools = [POOL_GRAPHEME_CONSONANT, POOL_PHONEME_CONSONANT, POOL_ALLOPHONE_CONSONANT]
    entropies_cons = []
    for content, pool in zip(contents, cons_pools):
        cons = [t for t in content if t in pool]
        c_cons = Counter(cons)
        c_cons_full = Counter({k: c_cons.get(k, 0) for k in pool})
        entropies_cons.append(evenness_stats(c_cons_full)["entropy_norm"])
    ax2.bar(labels, entropies_cons, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax2.set_ylabel("Normalized Shannon entropy (0–1)")
    ax2.set_title("Normalized Shannon entropy — consonant pool only (train_full, unseen=0)")
    ax2.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_dir / "evenness_consonant_only.png", dpi=150)
    plt.close()
    print(f"Saved plot: {out_dir / 'evenness_consonant_only.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
