import json
import os
from typing import Dict


def load_cache(cache_path: str) -> Dict[str, str]:
    """Load translation cache from JSON."""
    if not os.path.exists(cache_path):
        return {}

    with open(cache_path, "r", encoding="utf-8") as cache_file:
        try:
            data = json.load(cache_file)
        except json.JSONDecodeError:
            return {}

    if isinstance(data, dict):
        return {str(key): str(value) for key, value in data.items()}
    return {}


def save_cache(cache_path: str, cache: Dict[str, str]) -> None:
    """Save translation cache to JSON."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as cache_file:
        json.dump(cache, cache_file, ensure_ascii=False, indent=2)
