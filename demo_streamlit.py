import os
import time

import streamlit as st

from app import CACHE_FILE, GLOSSARY_FILE, translate_document
from cache_utils import load_cache, save_cache
from document_processor import split_sentences
from glossary import load_glossary
from translator_backend import get_model_registry


st.set_page_config(page_title="OpenNMT Local Demo", layout="wide")

BASE_DIR = os.environ.get("OPENNMT_PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SAVED_OUTPUT = os.path.join(OUTPUT_DIR, "streamlit_output.txt")


def run_demo_translation(
    text: str,
    model_type: str,
    glossary_enabled: bool,
    batch_size: int,
    beam_size: int,
):
    sentences = split_sentences(text)
    glossary = load_glossary(GLOSSARY_FILE) if glossary_enabled else {}
    cache = load_cache(CACHE_FILE)

    start_time = time.time()
    translations, cache_hits = translate_document(
        sentences=sentences,
        glossary=glossary,
        cache=cache,
        model_type=model_type,
        beam_size=beam_size,
        batch_size=batch_size,
        glossary_enabled=glossary_enabled,
        cache_enabled=True,
    )
    elapsed_time = time.time() - start_time
    save_cache(CACHE_FILE, cache)
    return "\n".join(translations), elapsed_time, cache_hits, len(sentences)


def main() -> None:
    model_registry = get_model_registry()

    st.title("OpenNMT Local Translation Demo")
    st.caption("Baseline model + small-scale finetuning + glossary + cache")

    with st.sidebar:
        model_type = st.selectbox("Model", ["baseline", "finetuned"], index=1)
        glossary_enabled = st.checkbox("Enable glossary", value=True)
        batch_size = st.selectbox("Batch size", [1, 4, 8], index=1)
        beam_size = st.selectbox("Beam size", [1, 3, 5], index=2)
        save_output = st.checkbox("Save output to txt", value=True)

        st.markdown("**Resolved models**")
        st.text(f"Baseline:\n{model_registry['baseline']}")
        st.text(f"Finetuned:\n{model_registry['finetuned']}")

    default_text = (
        "OpenNMT can be adapted to a small tea-domain dataset. "
        "The glossary keeps terms such as Longjing tea and green tea consistent. "
        "Batch translation improves throughput during local document translation."
    )
    source_text = st.text_area("Input English text", value=default_text, height=220)

    if st.button("Translate", type="primary"):
        if not source_text.strip():
            st.warning("Please enter some text.")
            return

        translated_text, elapsed_time, cache_hits, sentence_count = run_demo_translation(
            text=source_text,
            model_type=model_type,
            glossary_enabled=glossary_enabled,
            batch_size=batch_size,
            beam_size=beam_size,
        )

        st.subheader("Translation Output")
        st.text_area("Result", value=translated_text, height=220)

        col1, col2, col3 = st.columns(3)
        col1.metric("Elapsed Time", f"{elapsed_time:.2f} s")
        col2.metric("Sentences", sentence_count)
        col3.metric("Cache Hits", cache_hits)

        if save_output:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(SAVED_OUTPUT, "w", encoding="utf-8") as output_file:
                output_file.write(translated_text)
            st.success(f"Saved to {SAVED_OUTPUT}")


if __name__ == "__main__":
    main()
