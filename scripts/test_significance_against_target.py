#!/usr/bin/env python3
"""Statistical significance testing against frozen SOTA targets.

Uses one-sample Wilcoxon signed-rank test on 5-fold test SCC values.
H0: median SCC <= target
H1: median SCC > target
Reports p-value (one-tailed).

Also computes multi-seed robustness (CV) when multiple summaries provided.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_dataset_vs_target(
    fold_sccs: list[float], target: float, alpha: float = 0.05
) -> dict:
    arr = np.array(fold_sccs, dtype=np.float64)
    mean_scc = float(np.mean(arr))
    std_scc = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    diffs = arr - target
    n_above = int(np.sum(diffs > 0))

    if len(arr) < 2 or np.all(diffs == 0):
        p_value = 1.0
    else:
        try:
            _stat, p_value = wilcoxon(diffs, alternative="greater")
            p_value = float(p_value)
        except ValueError:
            p_value = 1.0

    return {
        "n_folds": len(arr),
        "mean_scc": mean_scc,
        "std_scc": std_scc,
        "target": target,
        "gap": mean_scc - target,
        "folds_above_target": n_above,
        "fold_sccs": fold_sccs,
        "p_value": p_value,
        "significant": p_value < alpha,
        "alpha": alpha,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Significance test vs frozen SOTA targets")
    p.add_argument(
        "--summary-jsons", nargs="+", required=True,
        help="Paths to SUMMARY.json files (one per seed)",
    )
    p.add_argument("--targets-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    targets = load_json(args.targets_json)
    summaries = [load_json(p_) for p_ in args.summary_jsons]
    n_seeds = len(summaries)

    TARGET_MAP = {
        "WT": "on_target_wt_scc",
        "ESP": "on_target_esp_scc",
        "HF": "on_target_hf_scc",
        "Sniper-Cas9": "on_target_sniper_cas9_scc",
        "HL60": "on_target_hl60_scc",
        "xCas9": None,
        "SpCas9-NG": None,
        "HCT116": None,
        "HELA": None,
    }

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_seeds": n_seeds,
        "alpha": args.alpha,
        "summary_paths": args.summary_jsons,
        "per_dataset": {},
        "mean9": {},
    }

    header = f"{'Dataset':<15} {'Target':<8} {'Mean SCC':<10} {'p-value':<10} {'Sig?':<5} {'Seeds CV':<10}"
    print(f"\nSignificance Tests (Wilcoxon, alpha={args.alpha}, {n_seeds} seed(s))")
    print(header)
    print("=" * len(header))

    all_dataset_means = []

    first_summary = summaries[0]
    ds_key = "dataset_summaries" if "dataset_summaries" in first_summary else "per_dataset_metrics"

    for ds_name in first_summary[ds_key]:
        target_key = TARGET_MAP.get(ds_name)
        target_val = targets.get(target_key) if target_key else None

        seed_mean_sccs = []
        all_fold_sccs = []
        for s in summaries:
            ds_data = s[ds_key][ds_name]
            mean_scc = ds_data.get("mean_scc", 0.0)
            seed_mean_sccs.append(mean_scc)
            folds = ds_data.get("folds", ds_data.get("per_fold_test_sccs", []))
            if isinstance(folds, list) and len(folds) > 0:
                if isinstance(folds[0], dict):
                    fold_sccs = [f["test_scc"] for f in folds]
                else:
                    fold_sccs = folds
                all_fold_sccs.extend(fold_sccs)

        grand_mean = float(np.mean(seed_mean_sccs))
        all_dataset_means.append(grand_mean)
        cv = float(np.std(seed_mean_sccs) / np.mean(seed_mean_sccs)) if n_seeds > 1 else 0.0

        ds_result = {
            "per_seed_means": seed_mean_sccs,
            "grand_mean": grand_mean,
            "cv_across_seeds": cv,
        }

        if target_val is not None and len(all_fold_sccs) >= 2:
            sig = test_dataset_vs_target(all_fold_sccs, target_val, args.alpha)
            ds_result.update(sig)
            sig_flag = "YES" if sig["significant"] else "NO"
            print(
                f"{ds_name:<15} {target_val:<8.5f} {grand_mean:<10.5f} "
                f"{sig['p_value']:<10.6f} {sig_flag:<5} {cv:<10.4f}"
            )
        else:
            print(f"{ds_name:<15} {'N/A':<8} {grand_mean:<10.5f} {'N/A':<10} {'N/A':<5} {cv:<10.4f}")

        results["per_dataset"][ds_name] = ds_result

    mean9_target = targets.get("on_target_mean9_scc")
    if mean9_target and len(all_dataset_means) >= 2:
        mean9 = float(np.mean(all_dataset_means))
        diffs = np.array(all_dataset_means) - mean9_target
        try:
            _stat, pval = wilcoxon(diffs, alternative="greater")
            pval = float(pval)
        except ValueError:
            pval = 1.0
        results["mean9"] = {
            "mean9_scc": mean9,
            "target": mean9_target,
            "gap": mean9 - mean9_target,
            "p_value": pval,
            "significant": pval < args.alpha,
            "per_dataset_means": all_dataset_means,
        }
        sig_flag = "YES" if pval < args.alpha else "NO"
        print(f"\n{'mean9':<15} {mean9_target:<8.5f} {mean9:<10.5f} {pval:<10.6f} {sig_flag:<5}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to: {args.output_json}")


if __name__ == "__main__":
    main()
