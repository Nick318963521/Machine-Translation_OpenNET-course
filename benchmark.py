import csv
import os
import time
from typing import Dict, List

from app import CACHE_FILE, GLOSSARY_FILE, INPUT_FILE, translate_document
from cache_utils import load_cache, save_cache
from document_processor import read_document, split_sentences
from glossary import load_glossary
from translator_backend import get_model_registry


BASE_DIR = os.environ.get("OPENNMT_PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
OUTPUT_CSV = os.path.join(BASE_DIR, "outputs", "benchmark_results.csv")


def benchmark_case(
    mode: str,
    model_type: str,
    model_path: str,
    sentences: List[str],
    batch_size: int,
    beam_size: int,
    glossary_enabled: bool,
    cache_enabled: bool,
) -> Dict[str, object]:
    glossary = load_glossary(GLOSSARY_FILE) if glossary_enabled else {}
    working_cache = load_cache(CACHE_FILE) if cache_enabled else {}

    start_time = time.perf_counter()
    _, cache_hits = translate_document(
        sentences=sentences,
        glossary=glossary,
        cache=working_cache,
        model_path=model_path,
        model_type=model_type,
        beam_size=beam_size,
        batch_size=batch_size,
        glossary_enabled=glossary_enabled,
        cache_enabled=cache_enabled,
    )
    elapsed_time = time.perf_counter() - start_time

    if cache_enabled:
        save_cache(CACHE_FILE, working_cache)

    sentence_count = len(sentences)
    return {
        "mode": mode,
        "model_type": model_type,
        "batch_size": batch_size,
        "beam_size": beam_size,
        "glossary_enabled": glossary_enabled,
        "cache_enabled": cache_enabled,
        "elapsed_time": round(elapsed_time, 6),
        "num_sentences": sentence_count,
        "sent_per_sec": round(sentence_count / elapsed_time, 4) if elapsed_time > 0 else 0.0,
        "cache_hits": cache_hits,
    }


def run_benchmark() -> List[Dict[str, object]]:
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    sentences = split_sentences(read_document(INPUT_FILE))
    repeated_sentences = sentences * 6
    model_registry = get_model_registry()
    results: List[Dict[str, object]] = []

    for model_type in ("baseline", "finetuned"):
        results.append(
            benchmark_case(
                mode="model_compare",
                model_type=model_type,
                model_path=model_registry[model_type],
                sentences=sentences,
                batch_size=4,
                beam_size=5,
                glossary_enabled=True,
                cache_enabled=False,
            )
        )

    for glossary_enabled in (False, True):
        results.append(
            benchmark_case(
                mode="glossary_compare",
                model_type="finetuned",
                model_path=model_registry["finetuned"],
                sentences=sentences,
                batch_size=4,
                beam_size=5,
                glossary_enabled=glossary_enabled,
                cache_enabled=False,
            )
        )

    for batch_size in (1, 4, 8):
        results.append(
            benchmark_case(
                mode="batch_compare",
                model_type="finetuned",
                model_path=model_registry["finetuned"],
                sentences=sentences,
                batch_size=batch_size,
                beam_size=5,
                glossary_enabled=True,
                cache_enabled=False,
            )
        )

    for beam_size in (1, 3, 5):
        results.append(
            benchmark_case(
                mode="beam_compare",
                model_type="finetuned",
                model_path=model_registry["finetuned"],
                sentences=sentences,
                batch_size=4,
                beam_size=beam_size,
                glossary_enabled=True,
                cache_enabled=False,
            )
        )

    for cache_enabled in (False, True):
        results.append(
            benchmark_case(
                mode="cache_compare",
                model_type="finetuned",
                model_path=model_registry["finetuned"],
                sentences=repeated_sentences,
                batch_size=4,
                beam_size=5,
                glossary_enabled=True,
                cache_enabled=cache_enabled,
            )
        )

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "mode",
                "model_type",
                "batch_size",
                "beam_size",
                "glossary_enabled",
                "cache_enabled",
                "elapsed_time",
                "num_sentences",
                "sent_per_sec",
                "cache_hits",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Benchmark results saved to: {OUTPUT_CSV}")
    return results


if __name__ == "__main__":
    for row in run_benchmark():
        print(row)
