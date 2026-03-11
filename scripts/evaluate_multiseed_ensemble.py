#!/usr/bin/env python3
"""Ensemble evaluation across multiple seeds for protocol-aligned HNN runs.

Reads per-fold results from each seed's SUMMARY.json, computes:
- per-dataset mean/std across seeds
- coefficient of variation (CV) as robustness indicator
- overall mean9 SCC across seeds
- robustness verdict per dataset (CV < 0.05 = robust, < 0.10 = acceptable)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-seed ensemble evaluation")
    p.add_argument("--summary-jsons", nargs="+", required=True)
    p.add_argument("--output-json", required=True)
    args = p.parse_args()

    summaries = [load_json(p_) for p_ in args.summary_jsons]
    n_seeds = len(summaries)

    first = summaries[0]
    ds_key = "dataset_summaries" if "dataset_summaries" in first else "per_dataset_metrics"
    dataset_names = list(first[ds_key].keys())

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_seeds": n_seeds,
        "summary_paths": args.summary_jsons,
        "per_dataset": {},
        "mean9_scc": None,
        "mean9_pcc": None,
    }

    all_ds_sccs = []
    all_ds_pccs = []

    print(f"\nMulti-Seed Ensemble ({n_seeds} seeds)")
    print(f"{'Dataset':<15} {'Mean SCC':<10} {'Std SCC':<10} {'CV':<8} {'Robust?'}")
    print("=" * 60)

    for ds_name in dataset_names:
        seed_sccs = []
        seed_pccs = []
        for s in summaries:
            ds_data = s[ds_key][ds_name]
            seed_sccs.append(ds_data["mean_scc"])
            seed_pccs.append(ds_data.get("mean_pcc", 0.0))

        mean_scc = float(np.mean(seed_sccs))
        std_scc = float(np.std(seed_sccs, ddof=1)) if n_seeds > 1 else 0.0
        mean_pcc = float(np.mean(seed_pccs))
        std_pcc = float(np.std(seed_pccs, ddof=1)) if n_seeds > 1 else 0.0
        cv = std_scc / mean_scc if mean_scc > 0 else float("inf")

        robust = "ROBUST" if cv < 0.05 else ("OK" if cv < 0.10 else "HIGH-VAR")

        results["per_dataset"][ds_name] = {
            "mean_scc": mean_scc,
            "std_scc": std_scc,
            "mean_pcc": mean_pcc,
            "std_pcc": std_pcc,
            "cv": cv,
            "robust": robust,
            "per_seed_sccs": seed_sccs,
            "per_seed_pccs": seed_pccs,
        }

        all_ds_sccs.append(mean_scc)
        all_ds_pccs.append(mean_pcc)
        print(f"{ds_name:<15} {mean_scc:<10.5f} {std_scc:<10.5f} {cv:<8.4f} {robust}")

    results["mean9_scc"] = float(np.mean(all_ds_sccs))
    results["mean9_pcc"] = float(np.mean(all_ds_pccs))

    print(f"\nOverall mean9 SCC: {results['mean9_scc']:.5f}")
    print(f"Overall mean9 PCC: {results['mean9_pcc']:.5f}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to: {args.output_json}")


if __name__ == "__main__":
    main()
