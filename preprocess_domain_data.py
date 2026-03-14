import csv
import os
import random
import re
from typing import List, Tuple


BASE_DIR = r"D:\opennmt_project"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_domain")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_CSV = os.path.join(RAW_DIR, "domain_pairs.csv")


def clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def tokenize_zh_text(text: str) -> str:
    """Use a simple character-level tokenizer for Chinese."""
    compact = re.sub(r"\s+", "", text.strip())
    if not compact:
        return ""
    return " ".join(list(compact))


def load_parallel_pairs(csv_path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            src = clean_text(row.get("source", ""))
            tgt = tokenize_zh_text(row.get("target", ""))
            if src and tgt:
                pairs.append((src, tgt))
    return pairs


def write_split(file_path: str, lines: List[str]) -> None:
    with open(file_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines))


def preprocess_domain_data(seed: int = 42) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    pairs = load_parallel_pairs(RAW_CSV)
    if len(pairs) < 10:
        raise ValueError("At least 10 parallel pairs are recommended for the demo split.")

    random.Random(seed).shuffle(pairs)
    total = len(pairs)
    train_end = max(1, int(total * 0.7))
    valid_end = max(train_end + 1, int(total * 0.85))

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

    print(f"Loaded {total} sentence pairs from {RAW_CSV}")
    print(f"Train: {len(train_pairs)}")
    print(f"Valid: {len(valid_pairs)}")
    print(f"Test: {len(test_pairs)}")
    print(f"Processed files saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    preprocess_domain_data()
