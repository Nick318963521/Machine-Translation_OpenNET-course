import csv
import re
from typing import Dict, Tuple


def load_glossary(csv_path: str) -> Dict[str, str]:
    """Load glossary as {source_term: target_term}."""
    glossary: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            source_term = row[0].strip()
            target_term = row[1].strip()
            if source_term and target_term:
                glossary[source_term] = target_term
    return glossary


def protect_terms(text: str, glossary: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
    """Replace glossary terms with placeholders before translation."""
    protected_text = text
    mapping: Dict[str, str] = {}
    sorted_terms = sorted(glossary.items(), key=lambda item: len(item[0]), reverse=True)

    for index, (source_term, target_term) in enumerate(sorted_terms):
        placeholder = f"__TERM_{index}__"
        pattern = re.compile(re.escape(source_term), flags=re.IGNORECASE)
        protected_text, count = pattern.subn(placeholder, protected_text)
        if count > 0:
            mapping[placeholder] = target_term

    return protected_text, mapping


def restore_terms(text: str, mapping: Dict[str, str]) -> str:
    """Restore placeholders with target-side glossary terms."""
    restored_text = text
    for placeholder, target_term in mapping.items():
        restored_text = restored_text.replace(placeholder, target_term)
    return restored_text
