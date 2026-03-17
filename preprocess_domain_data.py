import argparse
import csv
import os
import random
import re
from typing import List, Tuple


BASE_DIR = os.environ.get("OPENNMT_PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_domain")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_CSV = os.path.join(RAW_DIR, "domain_pairs.csv")


def clean_text(text: str) -> str:
    return " ".join(text.strip().split())


TOKEN_PATTERN = re.compile(r"\S+")


def token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def keep_pair(
    source: str,
    target: str,
    min_tokens: int,
    max_tokens: int,
    max_length_ratio: float,
) -> bool:
    src_len = token_count(source)
    tgt_len = token_count(target)
    if src_len < min_tokens or tgt_len < min_tokens:
        return False
    if src_len > max_tokens or tgt_len > max_tokens:
        return False
    short_len = min(src_len, tgt_len)
    long_len = max(src_len, tgt_len)
    if short_len == 0:
        return False
    return (long_len / short_len) <= max_length_ratio


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
    min_tokens: int = 2,
    max_tokens: int = 80,
    max_length_ratio: float = 3.0,
) -> None:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1).")
    if not 0 <= valid_ratio < 1:
        raise ValueError("valid_ratio must be in [0, 1).")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be < 1.")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    raw_pairs = load_parallel_pairs(csv_path, source_col=source_col, target_col=target_col)
    pairs = [
        (src, tgt)
        for src, tgt in raw_pairs
        if keep_pair(
            src,
            tgt,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            max_length_ratio=max_length_ratio,
        )
    ]

    dropped_by_filter = len(raw_pairs) - len(pairs)
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

    print(f"Loaded {len(raw_pairs)} sentence pairs from {csv_path}")
    print(f"Dropped by filters: {dropped_by_filter}")
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
    parser.add_argument("--min-tokens", type=int, default=2, help="Minimum token count per side.")
    parser.add_argument("--max-tokens", type=int, default=80, help="Maximum token count per side.")
    parser.add_argument(
        "--max-length-ratio",
        type=float,
        default=3.0,
        help="Maximum length ratio between source and target token counts.",
    )
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
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_length_ratio=args.max_length_ratio,
    )
