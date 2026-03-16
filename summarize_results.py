import argparse
import csv
import os
import re
from typing import Dict, List

from evaluate_bleu import compute_bleu


BASE_DIR = os.environ.get("OPENNMT_PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
BENCHMARK_CSV = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
REPORT_MD = os.path.join(OUTPUT_DIR, "course_report_summary.md")
COMPARE_CSV = os.path.join(OUTPUT_DIR, "course_report_metrics.csv")


def load_benchmark_rows(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def best_row(rows: List[Dict[str, str]], mode: str, metric: str, higher_is_better: bool) -> Dict[str, str]:
    candidates = [row for row in rows if row.get("mode") == mode]
    if not candidates:
        return {}
    key_fn = lambda row: float(row.get(metric, "0") or 0)
    return max(candidates, key=key_fn) if higher_is_better else min(candidates, key=key_fn)


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"[-+]?\d*\.?\d+", str(value))
        return float(match.group(0)) if match else 0.0


def pct_change(new_value: float, old_value: float) -> float:
    if old_value == 0:
        return 0.0
    return (new_value - old_value) / old_value * 100.0


def get_model_compare_stats(rows: List[Dict[str, str]]) -> Dict[str, float]:
    baseline = next(
        (row for row in rows if row.get("mode") == "model_compare" and row.get("model_type") == "baseline"),
        None,
    )
    finetuned = next(
        (row for row in rows if row.get("mode") == "model_compare" and row.get("model_type") == "finetuned"),
        None,
    )

    if not baseline or not finetuned:
        return {}

    baseline_bleu_speed = safe_float(baseline.get("sent_per_sec", "0"))
    finetuned_bleu_speed = safe_float(finetuned.get("sent_per_sec", "0"))
    baseline_time = safe_float(baseline.get("elapsed_time", "0"))
    finetuned_time = safe_float(finetuned.get("elapsed_time", "0"))

    return {
        "baseline_sent_per_sec": baseline_bleu_speed,
        "finetuned_sent_per_sec": finetuned_bleu_speed,
        "speed_change_pct": pct_change(finetuned_bleu_speed, baseline_bleu_speed),
        "baseline_elapsed_time": baseline_time,
        "finetuned_elapsed_time": finetuned_time,
        "latency_change_pct": pct_change(finetuned_time, baseline_time),
    }


def get_cache_compare_stats(rows: List[Dict[str, str]]) -> Dict[str, float]:
    cache_off = next(
        (row for row in rows if row.get("mode") == "cache_compare" and row.get("cache_enabled") == "False"),
        None,
    )
    cache_on = next(
        (row for row in rows if row.get("mode") == "cache_compare" and row.get("cache_enabled") == "True"),
        None,
    )

    if not cache_off or not cache_on:
        return {}

    off_speed = safe_float(cache_off.get("sent_per_sec", "0"))
    on_speed = safe_float(cache_on.get("sent_per_sec", "0"))
    off_time = safe_float(cache_off.get("elapsed_time", "0"))
    on_time = safe_float(cache_on.get("elapsed_time", "0"))
    on_hits = safe_float(cache_on.get("cache_hits", "0"))

    return {
        "cache_off_sent_per_sec": off_speed,
        "cache_on_sent_per_sec": on_speed,
        "cache_speedup_pct": pct_change(on_speed, off_speed),
        "cache_off_elapsed_time": off_time,
        "cache_on_elapsed_time": on_time,
        "cache_latency_reduction_pct": pct_change(off_time - on_time, off_time),
        "cache_hits": on_hits,
    }


def build_metrics_rows(
    baseline_bleu: float,
    finetuned_bleu: float,
    benchmark_rows: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    rows.append(
        {
            "category": "bleu",
            "metric": "baseline_bleu",
            "value": f"{baseline_bleu:.4f}",
            "notes": "test set",
        }
    )
    rows.append(
        {
            "category": "bleu",
            "metric": "finetuned_bleu",
            "value": f"{finetuned_bleu:.4f}",
            "notes": "test set",
        }
    )
    rows.append(
        {
            "category": "bleu",
            "metric": "bleu_improvement_pct",
            "value": f"{pct_change(finetuned_bleu, baseline_bleu):.2f}",
            "notes": "(finetuned-baseline)/baseline*100",
        }
    )

    model_stats = get_model_compare_stats(benchmark_rows)
    if model_stats:
        rows.append(
            {
                "category": "throughput",
                "metric": "model_compare_speed_change_pct",
                "value": f"{model_stats['speed_change_pct']:.2f}",
                "notes": "finetuned vs baseline sent_per_sec",
            }
        )
        rows.append(
            {
                "category": "latency",
                "metric": "model_compare_latency_change_pct",
                "value": f"{model_stats['latency_change_pct']:.2f}",
                "notes": "finetuned vs baseline elapsed_time",
            }
        )

    cache_stats = get_cache_compare_stats(benchmark_rows)
    if cache_stats:
        rows.append(
            {
                "category": "cache",
                "metric": "cache_speedup_pct",
                "value": f"{cache_stats['cache_speedup_pct']:.2f}",
                "notes": "cache_on vs cache_off sent_per_sec",
            }
        )
        rows.append(
            {
                "category": "cache",
                "metric": "cache_latency_reduction_pct",
                "value": f"{cache_stats['cache_latency_reduction_pct']:.2f}",
                "notes": "(off-on)/off*100",
            }
        )
        rows.append(
            {
                "category": "cache",
                "metric": "cache_hits",
                "value": f"{cache_stats['cache_hits']:.0f}",
                "notes": "cache_compare, cache_enabled=True",
            }
        )

    return rows


def save_metrics_csv(output_path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["category", "metric", "value", "notes"])
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(
    baseline_bleu: float,
    finetuned_bleu: float,
    benchmark_rows: List[Dict[str, str]],
    reference_path: str,
    baseline_pred_path: str,
    finetuned_pred_path: str,
    metrics_csv_path: str,
) -> str:
    best_batch = best_row(benchmark_rows, "batch_compare", "sent_per_sec", True)
    best_beam_speed = best_row(benchmark_rows, "beam_compare", "sent_per_sec", True)
    fastest_model = best_row(benchmark_rows, "model_compare", "sent_per_sec", True)
    cache_gain = best_row(benchmark_rows, "cache_compare", "cache_hits", True)

    lines: List[str] = []
    lines.append("# Course Project Result Summary")
    lines.append("")
    lines.append("## BLEU")
    lines.append(f"- Reference: {reference_path}")
    lines.append(f"- Baseline prediction: {baseline_pred_path}")
    lines.append(f"- Finetuned prediction: {finetuned_pred_path}")
    lines.append(f"- Baseline BLEU: {baseline_bleu:.4f}")
    lines.append(f"- Finetuned BLEU: {finetuned_bleu:.4f}")
    lines.append(f"- BLEU improvement: {pct_change(finetuned_bleu, baseline_bleu):.2f}%")
    lines.append("")
    lines.append("## Benchmark Highlights")

    if fastest_model:
        lines.append(
            "- Fastest model_compare run: "
            f"model_type={fastest_model['model_type']}, sent_per_sec={fastest_model['sent_per_sec']}"
        )
    if best_batch:
        lines.append(
            "- Best throughput in batch_compare: "
            f"batch_size={best_batch['batch_size']}, sent_per_sec={best_batch['sent_per_sec']}"
        )
    if best_beam_speed:
        lines.append(
            "- Fastest beam setting in beam_compare: "
            f"beam_size={best_beam_speed['beam_size']}, sent_per_sec={best_beam_speed['sent_per_sec']}"
        )
    if cache_gain:
        lines.append(
            "- Highest cache hit in cache_compare: "
            f"cache_enabled={cache_gain['cache_enabled']}, cache_hits={cache_gain['cache_hits']}"
        )

    model_stats = get_model_compare_stats(benchmark_rows)
    if model_stats:
        lines.append(
            "- Throughput change (finetuned vs baseline, model_compare): "
            f"{model_stats['speed_change_pct']:.2f}%"
        )

    cache_stats = get_cache_compare_stats(benchmark_rows)
    if cache_stats:
        lines.append(
            "- Cache speedup (cache_on vs cache_off): "
            f"{cache_stats['cache_speedup_pct']:.2f}%"
        )

    if not benchmark_rows:
        lines.append("- Benchmark data not found. Run benchmark.py first.")

    lines.append("")
    lines.append("## Tips for Course Report")
    lines.append("- Explain whether finetuning improved BLEU and by how much.")
    lines.append("- Discuss trade-offs between beam size and speed.")
    lines.append("- Discuss why cache improves repeated-sentence throughput.")
    lines.append(f"- Use metrics table from: {metrics_csv_path}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a markdown summary for the course report.")
    parser.add_argument("--reference", default=os.path.join(DATA_DIR, "test.tgt"))
    parser.add_argument("--baseline", default=os.path.join(OUTPUT_DIR, "baseline_test_pred.txt"))
    parser.add_argument("--finetuned", default=os.path.join(OUTPUT_DIR, "finetuned_test_pred.txt"))
    parser.add_argument("--benchmark", default=BENCHMARK_CSV)
    parser.add_argument("--output", default=REPORT_MD)
    parser.add_argument("--metrics-csv", default=COMPARE_CSV)
    args = parser.parse_args()

    baseline_bleu = safe_float(compute_bleu(args.reference, args.baseline))
    finetuned_bleu = safe_float(compute_bleu(args.reference, args.finetuned))
    rows = load_benchmark_rows(args.benchmark)
    metrics_rows = build_metrics_rows(baseline_bleu, finetuned_bleu, rows)

    markdown = build_markdown(
        baseline_bleu=baseline_bleu,
        finetuned_bleu=finetuned_bleu,
        benchmark_rows=rows,
        reference_path=args.reference,
        baseline_pred_path=args.baseline,
        finetuned_pred_path=args.finetuned,
        metrics_csv_path=args.metrics_csv,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output_file:
        output_file.write(markdown)
    save_metrics_csv(args.metrics_csv, metrics_rows)

    print(f"Saved summary report to: {args.output}")
    print(f"Saved metrics table to: {args.metrics_csv}")


if __name__ == "__main__":
    main()
