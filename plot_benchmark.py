import csv
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt


BASE_DIR = os.environ.get("OPENNMT_PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_CSV = os.path.join(BASE_DIR, "outputs", "benchmark_results.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def plot_batch_effect(rows: List[Dict[str, str]]) -> None:
    batch_rows = [row for row in rows if row["mode"] == "batch_compare"]
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in batch_rows:
        grouped[row["model_type"]].append(row)

    plt.figure(figsize=(8, 5))
    for model_type, group_rows in grouped.items():
        group_rows.sort(key=lambda row: int(row["batch_size"]))
        x_values = [int(row["batch_size"]) for row in group_rows]
        y_values = [float(row["elapsed_time"]) for row in group_rows]
        plt.plot(x_values, y_values, marker="o", label=model_type)

    plt.xlabel("Batch Size")
    plt.ylabel("Elapsed Time (s)")
    plt.title("Batch Size vs Elapsed Time")
    plt.xticks([1, 4, 8])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "batch_size_vs_time.png"), dpi=150)
    plt.close()


def plot_beam_effect(rows: List[Dict[str, str]]) -> None:
    beam_rows = [row for row in rows if row["mode"] == "beam_compare"]
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in beam_rows:
        grouped[row["model_type"]].append(row)

    plt.figure(figsize=(8, 5))
    for model_type, group_rows in grouped.items():
        group_rows.sort(key=lambda row: int(row["beam_size"]))
        x_values = [int(row["beam_size"]) for row in group_rows]
        y_values = [float(row["elapsed_time"]) for row in group_rows]
        plt.plot(x_values, y_values, marker="o", label=model_type)

    plt.xlabel("Beam Size")
    plt.ylabel("Elapsed Time (s)")
    plt.title("Beam Size vs Elapsed Time")
    plt.xticks([1, 3, 5])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "beam_size_vs_time.png"), dpi=150)
    plt.close()


def main() -> None:
    rows = load_rows(BENCHMARK_CSV)
    plot_batch_effect(rows)
    plot_beam_effect(rows)
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
