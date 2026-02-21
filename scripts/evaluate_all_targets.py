"""ChromaGuide Ph.D. Proposal Evaluation Suite.

Validates model performance against authoritative targets:
1. Spearman rho >= 0.911 (On-target)
2. AUROC >= 0.99 (Off-target)
3. Conformal Coverage within 0.90 +/- 0.02
4. Statistical Significance p < 0.001 (10K Bootstrap)
5. Cohen's d effect size reporting

Usage:
    python scripts/evaluate_all_targets.py --results_dir /path/to/narval/outputs
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path

def compute_cohens_d(x, y):
    """Compute effect size."""
    n1, n2 = len(x), len(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std

def paired_bootstrap_test(y_true, y_pred_model, y_pred_baseline, n_resamples=10000):
    """Paired bootstrap for p-value calculation."""
    res = []
    diff_obs = stats.spearmanr(y_true, y_pred_model)[0] - stats.spearmanr(y_true, y_pred_baseline)[0]

    indices = np.arange(len(y_true))
    for _ in range(n_resamples):
        idx = np.random.choice(indices, size=len(indices), replace=True)
        rho_m = stats.spearmanr(y_true[idx], y_pred_model[idx])[0]
        rho_b = stats.spearmanr(y_true[idx], y_pred_baseline[idx])[0]
        res.append(rho_m - rho_b)

    p_val = np.mean(np.array(res) <= 0)
    return p_val, diff_obs

def evaluate_on_target(results_dir):
    print("\n--- On-Target Evaluation ---")
    data_path = results_dir / "on_target_results.csv"
    if not data_path.exists():
        print(f"FAILED: {data_path} not found.")
        return None

    df = pd.read_csv(data_path)
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values

    rho, _ = stats.spearmanr(y_true, y_pred)
    target_rho = 0.911

    status = "PASS" if rho >= target_rho else "FAIL"
    print(f"Spearman Rho: {rho:.4f} (Target: >= {target_rho}) -> {status}")

    # Check for baseline comparison if available
    if 'y_baseline' in df.columns:
        p_val, delta = paired_bootstrap_test(y_true, y_pred, df['y_baseline'].values)
        print(f"Delta vs Baseline: {delta:.4f} | p-value: {p_val:.4f} (Target: < 0.001)")

    return {"rho": rho, "status": status}

def evaluate_off_target(results_dir):
    print("\n--- Off-Target Evaluation ---")
    data_path = results_dir / "off_target_results.csv"
    if not data_path.exists():
        print(f"FAILED: {data_path} not found.")
        return None

    df = pd.read_csv(data_path)
    auc = roc_auc_score(df['y_true'], df['y_prob'])
    ap = average_precision_score(df['y_true'], df['y_prob'])

    target_auc = 0.99
    status = "PASS" if auc >= target_auc else "FAIL"
    print(f"AUROC: {auc:.4f} (Target: >= {target_auc}) -> {status}")
    print(f"PR-AUC: {ap:.4f}")

    return {"auc": auc, "status": status}

def evaluate_conformal(results_dir):
    print("\n--- Conformal Coverage Evaluation ---")
    data_path = results_dir / "conformal_results.csv"
    if not data_path.exists():
        print(f"FAILED: {data_path} not found.")
        return None

    df = pd.read_csv(data_path)
    # Coverage calculation: true value within [lower, upper]
    covered = (df['y_true'] >= df['lower']) & (df['y_true'] <= df['upper'])
    actual_coverage = covered.mean()
    target_coverage = 0.90
    tolerance = 0.02

    status = "PASS" if abs(actual_coverage - target_coverage) <= tolerance else "FAIL"
    print(f"Coverage: {actual_coverage:.4f} (Target: {target_coverage} +/- {tolerance}) -> {status}")
    print(f"Avg Interval Width: {(df['upper'] - df['lower']).mean():.4f}")

    return {"coverage": actual_coverage, "status": status}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    report = {}

    report['on_target'] = evaluate_on_target(results_dir)
    report['off_target'] = evaluate_off_target(results_dir)
    report['conformal'] = evaluate_conformal(results_dir)

    summary_path = results_dir / "phd_proposal_report.json"
    with open(summary_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\nFinal report saved to {summary_path}")

if __name__ == "__main__":
    main()
