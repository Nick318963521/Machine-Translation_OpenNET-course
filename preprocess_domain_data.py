import argparse
import csv
import os
import random
from typing import List, Tuple


BASE_DIR = os.environ.get("OPENNMT_PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_domain")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_CSV = os.path.join(RAW_DIR, "domain_pairs.csv")


def clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def load_parallel_pairs(
    csv_path: str,
    source_col: str = "source",
    target_col: str = "target",
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        headers = reader.fieldnames or []
        if source_col not in headers or target_col not in headers:
            raise ValueError(
                f"CSV columns mismatch. Expected '{source_col}' and '{target_col}', got {headers}."
            )

        for row in reader:
            src = clean_text(row.get(source_col, ""))
            tgt = clean_text(row.get(target_col, ""))
            if src and tgt:
                pairs.append((src, tgt))
    return pairs


def write_split(file_path: str, lines: List[str]) -> None:
    with open(file_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines))


def preprocess_domain_data(
    csv_path: str = RAW_CSV,
    source_col: str = "source",
    target_col: str = "target",
    seed: int = 42,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    min_pairs: int = 10,
) -> None:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1).")
    if not 0 <= valid_ratio < 1:
        raise ValueError("valid_ratio must be in [0, 1).")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be < 1.")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    pairs = load_parallel_pairs(csv_path, source_col=source_col, target_col=target_col)
    if len(pairs) < min_pairs:
        raise ValueError(f"At least {min_pairs} parallel pairs are required.")

    random.Random(seed).shuffle(pairs)
    total = len(pairs)
    train_end = max(1, int(total * train_ratio))
    valid_end = max(train_end + 1, int(total * (train_ratio + valid_ratio)))

    train_pairs = pairs[:train_end]
    valid_pairs = pairs[train_end:valid_end]
    test_pairs = pairs[valid_end:]

    splits = {
        "train": train_pairs,
        "valid": valid_pairs,
        "test": test_pairs,
    }

    for split_name, split_pairs in splits.items():
        write_split(
            os.path.join(PROCESSED_DIR, f"{split_name}.src"),
            [src for src, _ in split_pairs],
        )
        write_split(
            os.path.join(PROCESSED_DIR, f"{split_name}.tgt"),
            [tgt for _, tgt in split_pairs],
        )

    print(f"Loaded {total} sentence pairs from {csv_path}")
    print(f"Columns: source='{source_col}', target='{target_col}'")
    print(f"Train: {len(train_pairs)}")
    print(f"Valid: {len(valid_pairs)}")
    print(f"Test: {len(test_pairs)}")
    print(f"Processed files saved to {PROCESSED_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess English-Spanish parallel data.")
    parser.add_argument("--csv", default=RAW_CSV, help="Path to parallel CSV.")
    parser.add_argument("--source-col", default="source", help="Source language column name.")
    parser.add_argument("--target-col", default="target", help="Target language column name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--valid-ratio", type=float, default=0.15, help="Valid split ratio.")
    parser.add_argument("--min-pairs", type=int, default=10, help="Minimum required pairs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_domain_data(
        csv_path=args.csv,
        source_col=args.source_col,
        target_col=args.target_col,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        min_pairs=args.min_pairs,
    )
