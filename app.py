import os
import sys
import time
from typing import Dict, List, Optional, Tuple

from cache_utils import load_cache, save_cache
from document_processor import read_document, split_sentences
from glossary import load_glossary, protect_terms, restore_terms
from translator_backend import get_model_registry, translate_sentences


BASE_DIR = os.environ.get("OPENNMT_PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

INPUT_FILE = os.path.join(DATA_DIR, "input.txt")
GLOSSARY_FILE = os.path.join(DATA_DIR, "glossary.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "translated.txt")
CACHE_FILE = os.path.join(CACHE_DIR, "translations.json")


def translate_document(
    sentences: List[str],
    glossary: Dict[str, str],
    cache: Dict[str, str],
    model_path: Optional[str] = None,
    model_type: str = "baseline",
    beam_size: int = 5,
    batch_size: int = 4,
    glossary_enabled: bool = True,
    cache_enabled: bool = True,
) -> Tuple[List[str], int]:
    results: List[str] = [""] * len(sentences)
    pending_inputs: List[str] = []
    pending_originals: List[str] = []
    pending_mappings: List[Dict[str, str]] = []
    pending_positions: Dict[str, List[int]] = {}
    cache_hits = 0

    for index, sentence in enumerate(sentences):
        if cache_enabled and sentence in cache:
            results[index] = cache[sentence]
            cache_hits += 1
            continue

        if cache_enabled and sentence in pending_positions:
            pending_positions[sentence].append(index)
            cache_hits += 1
            continue

        if glossary_enabled:
            protected_sentence, mapping = protect_terms(sentence, glossary)
        else:
            protected_sentence, mapping = sentence, {}

        pending_inputs.append(protected_sentence)
        pending_originals.append(sentence)
        pending_mappings.append(mapping)
        pending_positions[sentence] = [index]

    if pending_inputs:
        translated_batch = translate_sentences(
            pending_inputs,
            model_path=model_path,
            beam_size=beam_size,
            batch_size=batch_size,
            model_type=model_type,
        )
        for batch_index, translated_text in enumerate(translated_batch):
            restored_text = restore_terms(translated_text, pending_mappings[batch_index])
            original_sentence = pending_originals[batch_index]
            for sentence_index in pending_positions[original_sentence]:
                results[sentence_index] = restored_text
            if cache_enabled:
                cache[original_sentence] = restored_text

    return results, cache_hits


def run_app(
    input_file: str = INPUT_FILE,
    output_file: str = OUTPUT_FILE,
    model_type: str = "baseline",
    model_path: Optional[str] = None,
    beam_size: int = 5,
    batch_size: int = 4,
    glossary_enabled: bool = True,
    cache_enabled: bool = True,
) -> Tuple[List[str], float, int]:
    document_text = read_document(input_file)
    sentences = split_sentences(document_text)
    glossary = load_glossary(GLOSSARY_FILE) if glossary_enabled else {}
    cache = load_cache(CACHE_FILE) if cache_enabled else {}

    start_time = time.perf_counter()
    translations, cache_hits = translate_document(
        sentences=sentences,
        glossary=glossary,
        cache=cache,
        model_path=model_path,
        model_type=model_type,
        beam_size=beam_size,
        batch_size=batch_size,
        glossary_enabled=glossary_enabled,
        cache_enabled=cache_enabled,
    )
    elapsed_time = time.perf_counter() - start_time

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as output_handle:
        output_handle.write("\n".join(translations))

    if cache_enabled:
        save_cache(CACHE_FILE, cache)

    return translations, elapsed_time, cache_hits


def main() -> None:
    model_registry = get_model_registry()
    translations, elapsed_time, cache_hits = run_app(model_type="baseline")

    print("OpenNMT Local Translation Demo")
    print(f"Baseline model: {model_registry['baseline']}")
    print(f"Finetuned model candidate: {model_registry['finetuned']}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Cache hits: {cache_hits}")
    print("\nTranslation result:")
    for line in translations:
        safe_line = line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8", errors="replace"
        )
        print(safe_line)


if __name__ == "__main__":
    main()
