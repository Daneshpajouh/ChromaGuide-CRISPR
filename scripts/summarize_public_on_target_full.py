#!/usr/bin/env python3
"""Summarize completed public on-target full-run folds and compare to thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


DATASET_KEY_MAP = {
    "WT": "WT_SCC",
    "ESP": "ESP_SCC",
    "HF": "HF_SCC",
    "Sniper-Cas9": "Sniper_Cas9_SCC",
    "HL60": "HL60_SCC",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def collect_metrics(full_runs_dir: Path) -> dict[str, list[float]]:
    by_dataset: dict[str, list[float]] = {}
    for path in sorted(full_runs_dir.glob("*.json")):
        if path.name.startswith("SUMMARY_") or path.name.startswith("INTERIM_"):
            continue
        data = load_json(path)
        gold_rho = data.get("gold_rho")
        if not isinstance(gold_rho, (int, float)):
            continue
        dataset = path.stem.split("_fold")[0]
        by_dataset.setdefault(dataset, []).append(float(gold_rho))
    return by_dataset


def summarize(by_dataset: dict[str, list[float]], thresholds: dict) -> dict:
    out = {
        "completed_datasets": {},
        "mean_9_dataset_progress": None,
        "mean_9_dataset_claim_ready": False,
        "threshold_comparison": {},
    }
    completed_means = []
    avg_threshold = thresholds["on_target"]["average_thresholds"]["mean_SCC_9_dataset"]
    per_dataset_thresholds = thresholds["on_target"]["per_dataset_thresholds"]

    for dataset, values in sorted(by_dataset.items()):
        ds_mean = mean(values)
        completed_means.append(ds_mean)
        ds_summary = {
            "num_completed_folds": len(values),
            "mean_gold_rho": ds_mean,
            "stdev_gold_rho": pstdev(values) if len(values) > 1 else 0.0,
            "min_gold_rho": min(values),
            "max_gold_rho": max(values),
            "claim_valid_for_dataset": len(values) >= 5,
        }
        out["completed_datasets"][dataset] = ds_summary

        threshold_key = DATASET_KEY_MAP.get(dataset)
        if threshold_key:
            threshold_value = per_dataset_thresholds[threshold_key]
            out["threshold_comparison"][dataset] = {
                "threshold": threshold_value,
                "mean_gold_rho": ds_mean,
                "gap": ds_mean - threshold_value,
                "meets_threshold_now": ds_mean >= threshold_value and len(values) >= 5,
            }

    if completed_means:
        out["mean_9_dataset_progress"] = mean(completed_means)
        out["mean_9_dataset_gap_vs_hnn_target"] = out["mean_9_dataset_progress"] - avg_threshold
        out["mean_9_dataset_claim_ready"] = len(by_dataset) >= 9 and all(
            len(v) >= 5 for v in by_dataset.values()
        )
    else:
        out["mean_9_dataset_gap_vs_hnn_target"] = None

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional input directory. Defaults to results/public_benchmarks/full_runs under repo root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to results/public_benchmarks/full_runs/INTERIM_FULL_SUMMARY_2026-03-02.json",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    full_runs_dir = (repo_root / args.input_dir).resolve() if args.input_dir else (
        repo_root / "results" / "public_benchmarks" / "full_runs"
    )
    thresholds_path = repo_root / "public_claim_thresholds.json"
    output = args.output or (
        full_runs_dir / "INTERIM_FULL_SUMMARY_2026-03-02.json"
    )

    by_dataset = collect_metrics(full_runs_dir)
    thresholds = load_json(thresholds_path)
    result = {
        "generated": "2026-03-02",
        "source_dir": str(full_runs_dir),
        "thresholds_path": str(thresholds_path),
        "summary": summarize(by_dataset, thresholds),
    }
    output.write_text(json.dumps(result, indent=2))
    print(output)


if __name__ == "__main__":
    main()
