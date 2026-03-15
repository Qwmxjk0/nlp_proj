#!/usr/bin/env python3
"""Benchmark multiple Ollama models on the same Wongnai review sample."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/model_benchmarks")
DEFAULT_MODELS = [
    "scb10x/typhoon2.5-qwen3-4b",
    "scb10x/typhoon2.5-qwen3-30b-a3b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple Ollama models on identical samples.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model names to benchmark.")
    parser.add_argument("--sample-size", type=int, default=6, help="Number of rows to benchmark.")
    parser.add_argument("--limit-rows", type=int, default=100, help="Candidate pool size before sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--parallel", type=int, default=2, help="Parallel requests per benchmark run.")
    parser.add_argument("--num-ctx", type=int, default=2048, help="Context window.")
    parser.add_argument("--timeout", type=int, default=600, help="Request timeout in seconds.")
    parser.add_argument("--keep-alive", default="30m", help="Ollama keep_alive duration.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--base-url", default="http://localhost:11434/api/chat", help="Ollama chat endpoint.")
    parser.add_argument("--input", default="wongnai-review-dataset/review_dataset/w_review_train.csv", help="Input CSV path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Benchmark output directory.")
    return parser.parse_args()


def run_labeler(model: str, args: argparse.Namespace, model_output_dir: Path) -> tuple[int, float, str]:
    cmd = [
        sys.executable,
        "scripts/label_reviews_ollama.py",
        "--model",
        model,
        "--sample-size",
        str(args.sample_size),
        "--limit-rows",
        str(args.limit_rows),
        "--seed",
        str(args.seed),
        "--parallel",
        str(args.parallel),
        "--num-ctx",
        str(args.num_ctx),
        "--timeout",
        str(args.timeout),
        "--keep-alive",
        args.keep_alive,
        "--temperature",
        str(args.temperature),
        "--base-url",
        args.base_url,
        "--input",
        args.input,
        "--output-dir",
        str(model_output_dir),
    ]
    started = time.time()
    completed = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started
    return completed.returncode, elapsed, completed.stdout + "\n" + completed.stderr


def newest_subdir(parent: Path) -> Path | None:
    subdirs = [path for path in parent.iterdir() if path.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda path: path.stat().st_mtime)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_model(model: str, run_dir: Path, wall_time_sec: float, return_code: int) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json")
    perf = load_json(run_dir / "perf_report.json")
    return {
        "model": model,
        "return_code": return_code,
        "run_dir": str(run_dir),
        "wall_time_sec": round(wall_time_sec, 3),
        "success_rows": summary["success_count"],
        "error_rows": summary["error_count"],
        "latency_avg_sec": summary["latency_avg_sec"],
        "latency_median_sec": summary["latency_median_sec"],
        "latency_max_sec": summary["latency_max_sec"],
        "prompt_tokens_total": summary.get("prompt_tokens_total"),
        "completion_tokens_total": summary.get("completion_tokens_total"),
        "all_tokens_total": summary.get("all_tokens_total"),
        "completion_tokens_per_sec_run": summary.get("completion_tokens_per_sec_run"),
        "all_tokens_per_sec_run": summary.get("all_tokens_per_sec_run"),
        "successful_rows_per_sec_wall": perf["throughput"]["successful_rows_per_sec_wall"],
        "all_tokens_per_sec_wall": perf["throughput"]["all_tokens_per_sec_wall"],
    }


def write_report_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_decision_notes(rows: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    successful = [row for row in rows if row["return_code"] == 0]
    if not successful:
        return ["No successful benchmark runs completed."]
    fastest = max(successful, key=lambda row: row["successful_rows_per_sec_wall"] or 0)
    lowest_latency = min(successful, key=lambda row: row["latency_avg_sec"] or float("inf"))
    notes.append(
        f"Fastest wall-clock throughput: {fastest['model']} at {fastest['successful_rows_per_sec_wall']} rows/sec."
    )
    notes.append(
        f"Lowest average latency: {lowest_latency['model']} at {lowest_latency['latency_avg_sec']} sec/row."
    )
    if len(successful) >= 2:
        latencies = [row["latency_avg_sec"] for row in successful if row["latency_avg_sec"] is not None]
        if latencies:
            notes.append(
                f"Average latency spread across models: {round(min(latencies), 3)} to {round(max(latencies), 3)} sec/row."
            )
    return notes


def main() -> int:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    benchmark_dir = args.output_dir / timestamp
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    logs_dir = benchmark_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        safe_name = model.replace("/", "__").replace(":", "_")
        model_output_dir = benchmark_dir / safe_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Benchmarking {model} ...")
        return_code, wall_time_sec, output = run_labeler(model, args, model_output_dir)
        (logs_dir / f"{safe_name}.log").write_text(output, encoding="utf-8")

        run_dir = newest_subdir(model_output_dir)
        if run_dir is None:
            results.append(
                {
                    "model": model,
                    "return_code": return_code,
                    "run_dir": "",
                    "wall_time_sec": round(wall_time_sec, 3),
                    "success_rows": 0,
                    "error_rows": args.sample_size,
                    "latency_avg_sec": None,
                    "latency_median_sec": None,
                    "latency_max_sec": None,
                    "prompt_tokens_total": None,
                    "completion_tokens_total": None,
                    "all_tokens_total": None,
                    "completion_tokens_per_sec_run": None,
                    "all_tokens_per_sec_run": None,
                    "successful_rows_per_sec_wall": None,
                    "all_tokens_per_sec_wall": None,
                }
            )
            continue

        results.append(summarize_model(model, run_dir, wall_time_sec, return_code))

    summary = {
        "benchmark_dir": str(benchmark_dir),
        "models": args.models,
        "sample_size": args.sample_size,
        "limit_rows": args.limit_rows,
        "seed": args.seed,
        "parallel": args.parallel,
        "num_ctx": args.num_ctx,
        "notes": build_decision_notes(results),
        "results": results,
    }

    summary_path = benchmark_dir / "benchmark_summary.json"
    csv_path = benchmark_dir / "benchmark_summary.csv"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_report_csv(csv_path, results)

    print("")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary: {summary_path}")
    print(f"csv:     {csv_path}")
    print(f"logs:    {logs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
