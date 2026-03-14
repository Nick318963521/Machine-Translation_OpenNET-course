import re
from typing import List


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\r?\n+")


def read_document(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as input_file:
        return input_file.read().strip()


def split_sentences(text: str) -> List[str]:
    if not text.strip():
        return []
    return [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text.strip()) if part.strip()]
