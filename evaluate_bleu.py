import argparse
import os
import subprocess

from translator_backend import get_python_executable


BASE_DIR = os.environ.get("OPENNMT_PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def ensure_sacrebleu() -> None:
    try:
        import sacrebleu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "sacrebleu is required. Install it with: pip install sacrebleu"
        ) from exc


def compute_bleu(reference_path: str, prediction_path: str) -> str:
    ensure_sacrebleu()
    command = [
        get_python_executable(),
        "-m",
        "sacrebleu",
        reference_path,
        "-i",
        prediction_path,
        "-m",
        "bleu",
        "-b",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"sacreBLEU failed:\n{result.stderr}")
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline and finetuned BLEU.")
    parser.add_argument(
        "--reference",
        default=os.path.join(DATA_DIR, "test.tgt"),
        help="Reference target file.",
    )
    parser.add_argument(
        "--baseline",
        default=os.path.join(OUTPUT_DIR, "baseline_test_pred.txt"),
        help="Baseline prediction file.",
    )
    parser.add_argument(
        "--finetuned",
        default=os.path.join(OUTPUT_DIR, "finetuned_test_pred.txt"),
        help="Finetuned prediction file.",
    )
    args = parser.parse_args()

    baseline_bleu = compute_bleu(args.reference, args.baseline)
    finetuned_bleu = compute_bleu(args.reference, args.finetuned)

    print(f"Reference: {args.reference}")
    print(f"Baseline prediction: {args.baseline}")
    print(f"Finetuned prediction: {args.finetuned}")
    print(f"Baseline BLEU: {baseline_bleu}")
    print(f"Finetuned BLEU: {finetuned_bleu}")


if __name__ == "__main__":
    main()
