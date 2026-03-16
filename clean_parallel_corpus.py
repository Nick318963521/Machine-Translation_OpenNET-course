import argparse
import csv
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple


BASE_DIR = os.environ.get("OPENNMT_PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(BASE_DIR, "data", "raw_domain", "domain_pairs.csv")

MOJIBAKE_MARKERS = ("Ã", "Â", "â", "ð", "ï")
DROPPED_EXACT_SOURCES = {
    "translator-credits",
    "n/a",
}


def normalize_space(text: str) -> str:
    return " ".join((text or "").strip().split())


def strip_accelerator_underscore(text: str) -> str:
    # Common in localization corpora: menu accelerators such as _File, Ay_uda.
    return text.replace("_", "")


def remove_control_chars(text: str) -> str:
    return "".join(ch for ch in text if ch == "\t" or ord(ch) >= 32)


def maybe_fix_mojibake(text: str) -> str:
    if not text:
        return text

    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))

    # Path A: common latin1/utf8 mojibake such as "EspaÃ±a".
    if any(marker in text for marker in MOJIBAKE_MARKERS):
        try:
            fixed = text.encode("latin-1").decode("utf-8")
            if sum(m in fixed for m in MOJIBAKE_MARKERS) < sum(m in text for m in MOJIBAKE_MARKERS):
                text = fixed
                cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    # Path B: Chinese-looking mojibake from UTF-8 bytes decoded by GBK,
    # e.g. "F谩brica" -> "Fábrica".
    if cjk_count > 0:
        try:
            fixed_gbk = text.encode("gbk").decode("utf-8")
            fixed_cjk = len(re.findall(r"[\u4e00-\u9fff]", fixed_gbk))
            if fixed_cjk < cjk_count:
                return fixed_gbk
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    return text


def clean_pair(source: str, target: str) -> Tuple[str, str]:
    source = normalize_space(remove_control_chars(maybe_fix_mojibake(source)))
    target = normalize_space(remove_control_chars(maybe_fix_mojibake(target)))
    source = strip_accelerator_underscore(source)
    target = strip_accelerator_underscore(target)
    return source, target


def is_valid_pair(source: str, target: str) -> bool:
    if not source or not target:
        return False
    if source.lower() in DROPPED_EXACT_SOURCES:
        return False
    if len(source) < 2 or len(target) < 2:
        return False
    if source == target and len(source) <= 3:
        return False
    return True


def load_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_rows(csv_path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["source", "target"])
        writer.writeheader()
        writer.writerows(rows)


def clean_corpus(input_csv: str, output_csv: str, backup: bool = True) -> None:
    rows = load_rows(input_csv)
    total = len(rows)

    if backup and os.path.abspath(input_csv) == os.path.abspath(output_csv):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{input_csv}.bak_{ts}"
        with open(input_csv, "r", encoding="utf-8-sig", newline="") as src, open(
            backup_path, "w", encoding="utf-8", newline=""
        ) as dst:
            dst.write(src.read())
        print(f"Backup saved: {backup_path}")

    cleaned: List[Dict[str, str]] = []
    seen = set()
    dropped_invalid = 0
    dropped_dup = 0
    modified = 0

    for row in rows:
        src_raw = row.get("source", "")
        tgt_raw = row.get("target", "")
        src, tgt = clean_pair(src_raw, tgt_raw)

        if (src, tgt) != (src_raw, tgt_raw):
            modified += 1

        if not is_valid_pair(src, tgt):
            dropped_invalid += 1
            continue

        key = (src, tgt)
        if key in seen:
            dropped_dup += 1
            continue
        seen.add(key)
        cleaned.append({"source": src, "target": tgt})

    write_rows(output_csv, cleaned)

    print(f"Input rows: {total}")
    print(f"Output rows: {len(cleaned)}")
    print(f"Modified rows: {modified}")
    print(f"Dropped invalid rows: {dropped_invalid}")
    print(f"Dropped duplicate rows: {dropped_dup}")
    print(f"Written: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and normalize English-Spanish parallel corpus CSV.")
    parser.add_argument("--input", default=RAW_CSV, help="Input CSV path.")
    parser.add_argument(
        "--output",
        default=RAW_CSV,
        help="Output CSV path. Default overwrites input after backup.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable backup when overwriting input file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean_corpus(args.input, args.output, backup=not args.no_backup)
