import glob
import os
import subprocess
import uuid
from typing import Dict, List, Optional


BASE_DIR = os.environ.get("OPENNMT_PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
OPENNMT_DIR = os.path.join(BASE_DIR, "OpenNMT-py")
TMP_DIR = os.path.join(BASE_DIR, "tmp")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRANSLATE_SCRIPT = os.path.join(OPENNMT_DIR, "translate.py")

BASELINE_MODEL_CANDIDATES = [
    os.path.join(MODELS_DIR, "baseline_pretrained.pt"),
    os.path.join(OPENNMT_DIR, "onmt", "tests", "test_model.pt"),
]

FINETUNED_MODEL_GLOB_PATTERNS = [
    os.path.join(MODELS_DIR, "averaged-10-epoch.pt"),
    os.path.join(MODELS_DIR, "finetuned_model.pt"),
    os.path.join(MODELS_DIR, "finetuned_step_*.pt"),
    os.path.join(MODELS_DIR, "finetuned_model_step_*.pt"),
    os.path.join(MODELS_DIR, "model_step_*.pt"),
]


def get_python_executable() -> str:
    candidates = [
        os.path.join(BASE_DIR, "venv", "Scripts", "python.exe"),
        os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe"),
        "python",
    ]
    for candidate in candidates:
        try:
            probe = subprocess.run(
                [candidate, "-c", "import fasttext"],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode == 0:
                return candidate
        except FileNotFoundError:
            continue
    return "python"


def resolve_model_path(model_path: Optional[str] = None, model_type: str = "baseline") -> str:
    if model_path and os.path.exists(model_path):
        return model_path

    if model_type == "baseline":
        for candidate in BASELINE_MODEL_CANDIDATES:
            if os.path.exists(candidate):
                return candidate
        matched_files: List[str] = []
        for pattern in FINETUNED_MODEL_GLOB_PATTERNS:
            matched_files.extend(glob.glob(pattern))
        matched_files = [path for path in matched_files if os.path.exists(path)]
        if matched_files:
            matched_files.sort(key=os.path.getmtime, reverse=True)
            return matched_files[0]
        raise FileNotFoundError("Baseline model not found.")

    if model_type == "finetuned":
        matched_files: List[str] = []
        for pattern in FINETUNED_MODEL_GLOB_PATTERNS:
            matched_files.extend(glob.glob(pattern))
        matched_files = [path for path in matched_files if os.path.exists(path)]
        if matched_files:
            matched_files.sort(key=os.path.getmtime, reverse=True)
            return matched_files[0]
        for fallback in BASELINE_MODEL_CANDIDATES:
            if os.path.exists(fallback):
                return fallback
        raise FileNotFoundError("Finetuned model not found, and baseline fallback is unavailable.")

    raise ValueError(f"Unsupported model_type: {model_type}")


def run_translation(
    model_path: str,
    src_file: str,
    output_file: str,
    beam_size: int = 5,
    batch_size: int = 4,
) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(TRANSLATE_SCRIPT):
        command = [
            get_python_executable(),
            TRANSLATE_SCRIPT,
            "-model",
            model_path,
            "-src",
            src_file,
            "-output",
            output_file,
            "-beam_size",
            str(beam_size),
            "-batch_size",
            str(batch_size),
            "-gpu",
            "-1",
        ]
        cwd = OPENNMT_DIR
    else:
        command = [
            get_python_executable(),
            "-m",
            "onmt.bin.translate",
            "-model",
            model_path,
            "-src",
            src_file,
            "-output",
            output_file,
            "-beam_size",
            str(beam_size),
            "-batch_size",
            str(batch_size),
            "-gpu",
            "-1",
        ]
        cwd = BASE_DIR

    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "OpenNMT translation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def translate_sentences(
    sentences: List[str],
    model_path: Optional[str] = None,
    beam_size: int = 5,
    batch_size: int = 4,
    model_type: str = "baseline",
) -> List[str]:
    if not sentences:
        return []

    resolved_model_path = resolve_model_path(model_path=model_path, model_type=model_type)
    os.makedirs(TMP_DIR, exist_ok=True)
    token = uuid.uuid4().hex
    src_file = os.path.join(TMP_DIR, f"src_{token}.txt")
    out_file = os.path.join(TMP_DIR, f"pred_{token}.txt")

    try:
        with open(src_file, "w", encoding="utf-8") as src_handle:
            src_handle.write("\n".join(sentences))

        run_translation(
            model_path=resolved_model_path,
            src_file=src_file,
            output_file=out_file,
            beam_size=beam_size,
            batch_size=batch_size,
        )

        with open(out_file, "r", encoding="utf-8") as out_handle:
            predictions = [line.rstrip("\n") for line in out_handle]

        if len(predictions) != len(sentences):
            raise ValueError(
                f"Prediction count mismatch: expected {len(sentences)}, got {len(predictions)}."
            )
        return predictions
    finally:
        for temp_file in (src_file, out_file):
            if os.path.exists(temp_file):
                os.remove(temp_file)


def get_model_registry() -> Dict[str, str]:
    baseline = resolve_model_path(model_type="baseline")
    finetuned = resolve_model_path(model_type="finetuned")
    return {"baseline": baseline, "finetuned": finetuned}
