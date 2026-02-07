"""
KSS dataset metadata split script.

Reads transcript.v.1.4.txt and produces:
- Test set: random 100 lines -> test.txt (strict exclusion from all train sets).
- Train subsets (cumulative): train_1h.txt, train_3h.txt, train_5h.txt, train_full.txt.

Input format: filepath|original|pronunciation|duration_sec|...
"""

import argparse
import random
from pathlib import Path


# Field index (0-based). KSS transcript.v.1.4: path|orig|orig|decomposed|duration_sec|english -> 5th field = index 4
FIELD_DURATION_IDX = 4
TEST_SIZE = 100
SEED = 42

# Cumulative duration targets (seconds)
TARGET_1H = 3600
TARGET_3H = 10800
TARGET_5H = 18000


def parse_line(line: str) -> tuple[str, float] | None:
    """Parse one line; return (original_line, duration_sec) or None if invalid."""
    line = line.strip()
    if not line:
        return None
    parts = line.split("|")
    if len(parts) <= FIELD_DURATION_IDX:
        return None
    try:
        duration = float(parts[FIELD_DURATION_IDX].strip())
    except (ValueError, TypeError):
        return None
    if duration <= 0:
        return None
    return (line, duration)


def get_file_id(line: str) -> str:
    """Extract file id (path) from one metadata line."""
    return line.split("|", 1)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split KSS transcript.v.1.4.txt into Train/Test and duration-based train subsets."
    )
    parser.add_argument(
        "transcript",
        type=Path,
        default=Path("data/KSS/transcript.v.1.4.txt"),
        nargs="?",
        help="Path to transcript.v.1.4.txt",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as transcript file)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    args = parser.parse_args()

    transcript_path = args.transcript
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    out_dir = args.out_dir or transcript_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Parse all lines (only those with valid duration)
    records: list[tuple[str, float]] = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is not None:
                records.append(parsed)

    if len(records) < TEST_SIZE:
        raise ValueError(
            f"Valid lines ({len(records)}) < Test size ({TEST_SIZE}). "
            "Check that duration is in the 5th field (index 4)."
        )

    rng = random.Random(args.seed)

    # 2) Test set: sample exactly 100 at random
    indices = list(range(len(records)))
    rng.shuffle(indices)
    test_indices = set(indices[:TEST_SIZE])
    train_indices = [i for i in indices[TEST_SIZE:]]

    test_lines = [records[i][0] for i in sorted(test_indices)]
    test_file_ids = {get_file_id(ln) for ln in test_lines}

    test_path = out_dir / "test.txt"
    with open(test_path, "w", encoding="utf-8") as f:
        for ln in test_lines:
            f.write(ln + "\n")
    print(f"[Test] Saved: {test_path}")
    print(f"  - Sentences: {len(test_lines)}")
    print(f"  - Cumulative duration: {sum(records[i][1] for i in test_indices):.1f} sec")

    # 3) Shuffle remainder and build cumulative train subsets
    rng.shuffle(train_indices)
    train_records_ordered = [(records[i][0], records[i][1]) for i in train_indices]

    def write_subset(
        name: str,
        lines_with_dur: list[tuple[str, float]],
        path: Path,
    ) -> float:
        total_sec = sum(d for _, d in lines_with_dur)
        with open(path, "w", encoding="utf-8") as f:
            for ln, _ in lines_with_dur:
                f.write(ln + "\n")
        print(f"[{name}] Saved: {path}")
        print(f"  - Sentences: {len(lines_with_dur)}")
        print(f"  - Cumulative duration: {total_sec:.1f} sec ({total_sec / 60:.1f} min)")
        return total_sec

    cumulative_lines: list[tuple[str, float]] = []
    cumulative_sec = 0.0

    # train_1h
    for line, dur in train_records_ordered:
        cumulative_lines.append((line, dur))
        cumulative_sec += dur
        if cumulative_sec >= TARGET_1H:
            break
    write_subset("train_1h", cumulative_lines.copy(), out_dir / "train_1h.txt")
    n_1h = len(cumulative_lines)

    # train_3h
    for line, dur in train_records_ordered[n_1h:]:
        cumulative_lines.append((line, dur))
        cumulative_sec += dur
        if cumulative_sec >= TARGET_3H:
            break
    write_subset("train_3h", cumulative_lines.copy(), out_dir / "train_3h.txt")
    n_3h = len(cumulative_lines)

    # train_5h
    for line, dur in train_records_ordered[n_3h:]:
        cumulative_lines.append((line, dur))
        cumulative_sec += dur
        if cumulative_sec >= TARGET_5H:
            break
    write_subset("train_5h", cumulative_lines.copy(), out_dir / "train_5h.txt")
    n_5h = len(cumulative_lines)

    # train_full: all data except test
    train_full = [(records[i][0], records[i][1]) for i in train_indices]
    write_subset("train_full", train_full, out_dir / "train_full.txt")

    # 4) Validation: no test file must appear in any train set
    for name, fname in [
        ("train_1h", "train_1h.txt"),
        ("train_3h", "train_3h.txt"),
        ("train_5h", "train_5h.txt"),
        ("train_full", "train_full.txt"),
    ]:
        path = out_dir / fname
        with open(path, "r", encoding="utf-8") as f:
            train_ids = {get_file_id(ln) for ln in f if ln.strip()}
        overlap = test_file_ids & train_ids
        assert not overlap, (
            f"Validation failed: {name} contains test set files: {overlap}"
        )
        print(f"  [Validation] {name}: no overlap with test set (OK)")

    print("\nSplit and validation completed.")


if __name__ == "__main__":
    main()
