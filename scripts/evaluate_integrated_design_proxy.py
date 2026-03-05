#!/usr/bin/env python3
"""Evaluate integrated sgRNA ranking against on-target-only and off-target-only baselines.

This benchmark is proxy-based because a single dataset with matched on-target efficacy
and per-guide off-target risk labels is not available in this workspace.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import ndcg_score
from sklearn.model_selection import KFold, train_test_split


def max_homopolymer_run(seq: str) -> int:
    if not isinstance(seq, str) or not seq:
        return 0
    best = 1
    cur = 1
    s = seq.upper()
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best


def frac(seq: str, base: str) -> float:
    if not isinstance(seq, str) or not seq:
        return 0.0
    s = seq.upper()
    return float(s.count(base) / len(s))


def seq_feature_frame(seqs: pd.Series) -> pd.DataFrame:
    rows = []
    for s in seqs.astype(str).tolist():
        a = frac(s, "A")
        c = frac(s, "C")
        g = frac(s, "G")
        t = frac(s, "T")
        gc = g + c
        gg = float("GG" in s)
        tt = float("TTTT" in s)
        run = max_homopolymer_run(s)
        rows.append(
            {
                "a_frac": a,
                "c_frac": c,
                "g_frac": g,
                "t_frac": t,
                "gc_frac": gc,
                "gg_flag": gg,
                "poly_t_flag": tt,
                "max_run": float(run),
            }
        )
    return pd.DataFrame(rows)


def zscore(x: np.ndarray) -> np.ndarray:
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if sd < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mu) / sd


def ndcg_at_k(y_true_util: np.ndarray, y_score: np.ndarray, k: int) -> float:
    rel = y_true_util - float(np.min(y_true_util))
    rel = rel + 1e-8
    return float(ndcg_score(rel.reshape(1, -1), y_score.reshape(1, -1), k=k))


def topk_mean_utility(y_true_util: np.ndarray, y_score: np.ndarray, k: int) -> float:
    order = np.argsort(-y_score)
    k_eff = min(k, len(order))
    return float(np.mean(y_true_util[order[:k_eff]]))


@dataclass
class ScoreMetrics:
    spearman: float
    ndcg10: float
    ndcg20: float
    top10_mean_utility: float
    top20_mean_utility: float


def compute_metrics(y_util: np.ndarray, score: np.ndarray) -> ScoreMetrics:
    rho = float(spearmanr(y_util, score).statistic)
    return ScoreMetrics(
        spearman=rho,
        ndcg10=ndcg_at_k(y_util, score, 10),
        ndcg20=ndcg_at_k(y_util, score, 20),
        top10_mean_utility=topk_mean_utility(y_util, score, 10),
        top20_mean_utility=topk_mean_utility(y_util, score, 20),
    )


def bootstrap_delta(
    y_util: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y_util))
    d_top20 = np.zeros(n_boot, dtype=np.float64)
    d_rho = np.zeros(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        ua = topk_mean_utility(y_util[idx], s_a[idx], 20)
        ub = topk_mean_utility(y_util[idx], s_b[idx], 20)
        d_top20[i] = ua - ub
        d_rho[i] = float(spearmanr(y_util[idx], s_a[idx]).statistic - spearmanr(y_util[idx], s_b[idx]).statistic)
    return {
        "delta_top20_mean_utility": float(np.mean(d_top20)),
        "delta_spearman": float(np.mean(d_rho)),
        "top20_ci95": [float(np.percentile(d_top20, 2.5)), float(np.percentile(d_top20, 97.5))],
        "spearman_ci95": [float(np.percentile(d_rho, 2.5)), float(np.percentile(d_rho, 97.5))],
        "p_delta_top20_le_0": float(np.mean(d_top20 <= 0)),
        "p_delta_spearman_le_0": float(np.mean(d_rho <= 0)),
    }


def load_top_prediction_paths(results_dir: str, max_models: int) -> tuple[list[str], list[float]]:
    records: list[tuple[float, str, str]] = []
    for jp in glob.glob(os.path.join(results_dir, "*.json")):
        name = os.path.basename(jp)
        if any(x in name for x in ("summary", "submitted", "significance", "conformal", "offtarget", "ensemble")):
            continue
        if any(x in name for x in ("oof", "stack", "meta", "rankfusion")):
            continue
        try:
            with open(jp, "r") as f:
                d = json.load(f)
        except Exception:
            continue
        rho = d.get("gold_rho")
        if rho is None:
            continue
        pp = d.get("predictions_file")
        if not pp:
            pp = jp.replace(".json", "_predictions.csv")
        if not os.path.isabs(pp):
            pp = os.path.join(results_dir, os.path.basename(pp))
        if not os.path.exists(pp):
            continue
        records.append((float(rho), pp, name))
    records.sort(key=lambda x: x[0], reverse=True)
    chosen = records[:max_models]
    return [p for _, p, _ in chosen], [r for r, _, _ in chosen]


def main() -> None:
    ap = argparse.ArgumentParser(description="Proxy integrated ranking benchmark")
    ap.add_argument("--pred_csv", default="results/runs/w33_oof_hgbr_k240_d8_i300_lr004_s21_predictions.csv")
    ap.add_argument("--benchmark_csv", default="data/real/benchmark_merged.csv")
    ap.add_argument("--results_dir", default="results/runs")
    ap.add_argument("--max_models_for_sigma", type=int, default=120)
    ap.add_argument("--utility_risk_weight", type=float, default=0.6)
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--output_json", default="results/runs/w36_integrated_design_proxy_eval.json")
    ap.add_argument("--output_ranked_csv", default="results/runs/w36_integrated_design_proxy_ranked.csv")
    args = ap.parse_args()

    base = pd.read_csv(args.pred_csv)
    if not {"sequence", "efficiency", "prediction"}.issubset(base.columns):
        raise ValueError("pred_csv must include columns: sequence, efficiency, prediction")
    base = base[["sequence", "efficiency", "prediction"]].copy()
    base = base.rename(columns={"efficiency": "y_true_eff", "prediction": "mu_pred"})

    bench = pd.read_csv(args.benchmark_csv)
    feat_cols = [c for c in bench.columns if c.startswith("feat_")]
    agg_map = {c: "mean" for c in feat_cols}
    feat_by_seq = bench.groupby("sequence", as_index=False).agg(agg_map)
    df = base.merge(feat_by_seq, on="sequence", how="left")

    sf = seq_feature_frame(df["sequence"])
    df = pd.concat([df.reset_index(drop=True), sf.reset_index(drop=True)], axis=1)

    if "feat_0" in df.columns:
        df["gc_proxy"] = df["feat_0"].fillna(df["gc_frac"])
    else:
        df["gc_proxy"] = df["gc_frac"]
    if "feat_2" in df.columns:
        df["t_proxy"] = df["feat_2"].fillna(df["t_frac"])
    else:
        df["t_proxy"] = df["t_frac"]
    if "feat_10" in df.columns:
        f10 = df["feat_10"].fillna(df["feat_10"].median() if df["feat_10"].notna().any() else 0.0).values
        df["f10_norm"] = zscore(f10)
    else:
        df["f10_norm"] = 0.0

    gc_dist = np.abs(df["gc_proxy"].values - 0.5)
    run_norm = np.minimum(df["max_run"].values / 6.0, 1.0)
    risk_logit = 2.2 * gc_dist + 1.1 * run_norm + 0.9 * df["t_proxy"].values + 0.35 * df["gg_flag"].values - 1.0
    df["risk_true_proxy"] = 1.0 / (1.0 + np.exp(-risk_logit))

    x_cols = [
        "a_frac",
        "c_frac",
        "g_frac",
        "t_frac",
        "gc_frac",
        "gg_flag",
        "poly_t_flag",
        "max_run",
        "f10_norm",
    ]
    x = df[x_cols].values.astype(np.float64)
    y_risk = df["risk_true_proxy"].values.astype(np.float64)

    oof_risk = np.zeros(len(df), dtype=np.float64)
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    for i, (tr, va) in enumerate(kf.split(x), start=1):
        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=350,
            random_state=args.seed + i,
        )
        model.fit(x[tr], y_risk[tr])
        oof_risk[va] = model.predict(x[va])
    df["risk_pred"] = np.clip(oof_risk, 0.0, 1.0)

    pred_paths, pred_rhos = load_top_prediction_paths(args.results_dir, args.max_models_for_sigma)
    sigma_mat = []
    for p in pred_paths:
        try:
            d = pd.read_csv(p, usecols=["sequence", "prediction"])
        except Exception:
            continue
        col = os.path.basename(p).replace(".csv", "")
        d = d.rename(columns={"prediction": col})
        m = df[["sequence"]].merge(d, on="sequence", how="left")
        miss = float(m[col].isna().mean())
        if miss > 0.05:
            continue
        sigma_mat.append(m[col].fillna(m[col].mean()).values.astype(np.float64))
    if sigma_mat:
        sigma_arr = np.vstack(sigma_mat)
        df["sigma_pred"] = np.std(sigma_arr, axis=0)
    else:
        df["sigma_pred"] = np.abs(df["mu_pred"] - float(df["mu_pred"].mean()))

    y_util = zscore(df["y_true_eff"].values) - args.utility_risk_weight * zscore(df["risk_true_proxy"].values)
    df["utility_true"] = y_util

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed, shuffle=True)
    tr_mask = np.zeros(len(df), dtype=bool)
    tr_mask[tr_idx] = True
    te_mask = ~tr_mask

    mu = df["mu_pred"].values.astype(np.float64)
    risk = df["risk_pred"].values.astype(np.float64)
    mu_z = zscore(mu)
    risk_z = zscore(risk)
    sig = zscore(df["sigma_pred"].values.astype(np.float64))

    best_wr, best_wu = 0.5, 0.2
    best_obj = -math.inf
    wr_grid = [0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.1, 1.4, 1.8, 2.2]
    wu_grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    for wr in wr_grid:
        for wu in wu_grid:
            s = mu_z[tr_mask] - wr * risk_z[tr_mask] - wu * sig[tr_mask]
            obj = ndcg_at_k(y_util[tr_mask], s, 20)
            if obj > best_obj:
                best_obj = obj
                best_wr = wr
                best_wu = wu

    df["score_on_target_only"] = mu
    df["score_off_target_only"] = -risk_z
    df["score_integrated"] = mu_z - best_wr * risk_z - best_wu * sig
    df["split"] = np.where(te_mask, "test", "train")

    y_test = y_util[te_mask]
    s_on = df.loc[te_mask, "score_on_target_only"].values.astype(np.float64)
    s_off = df.loc[te_mask, "score_off_target_only"].values.astype(np.float64)
    s_int = df.loc[te_mask, "score_integrated"].values.astype(np.float64)

    m_on = compute_metrics(y_test, s_on)
    m_off = compute_metrics(y_test, s_off)
    m_int = compute_metrics(y_test, s_int)

    b_int_vs_on = bootstrap_delta(y_test, s_int, s_on, args.bootstrap, args.seed + 100)
    b_int_vs_off = bootstrap_delta(y_test, s_int, s_off, args.bootstrap, args.seed + 200)

    report = {
        "benchmark_type": "proxy_integrated_ranking",
        "notes": "Utility uses measured on-target efficacy with sequence-derived off-target proxy due missing matched real guide-level off-target labels in this workspace.",
        "inputs": {
            "pred_csv": args.pred_csv,
            "benchmark_csv": args.benchmark_csv,
            "results_dir": args.results_dir,
            "rows_total": int(len(df)),
            "rows_train": int(tr_mask.sum()),
            "rows_test": int(te_mask.sum()),
            "top_models_for_sigma": int(len(sigma_mat)),
            "top_model_rho_min": float(min(pred_rhos)) if pred_rhos else None,
            "top_model_rho_max": float(max(pred_rhos)) if pred_rhos else None,
        },
        "utility_definition": {
            "formula": "z(efficiency_true) - alpha * z(risk_true_proxy)",
            "alpha": float(args.utility_risk_weight),
        },
        "integrated_score_definition": {
            "formula": "z(mu_pred) - w_r * z(risk_pred) - w_u * z(sigma_pred)",
            "w_r": float(best_wr),
            "w_u": float(best_wu),
            "selection_objective": "train ndcg@20",
            "train_best_ndcg20": float(best_obj),
        },
        "test_metrics": {
            "on_target_only": m_on.__dict__,
            "off_target_only": m_off.__dict__,
            "integrated": m_int.__dict__,
        },
        "bootstrap_integrated_vs_on_target_only": b_int_vs_on,
        "bootstrap_integrated_vs_off_target_only": b_int_vs_off,
        "passes": {
            "integrated_ndcg20_gt_on_target_only": bool(m_int.ndcg20 > m_on.ndcg20),
            "integrated_ndcg20_gt_off_target_only": bool(m_int.ndcg20 > m_off.ndcg20),
            "integrated_top20_utility_gt_on_target_only": bool(m_int.top20_mean_utility > m_on.top20_mean_utility),
            "integrated_top20_utility_gt_off_target_only": bool(m_int.top20_mean_utility > m_off.top20_mean_utility),
            "integrated_spearman_gt_on_target_only": bool(m_int.spearman > m_on.spearman),
            "integrated_spearman_gt_off_target_only": bool(m_int.spearman > m_off.spearman),
        },
    }

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)

    ranked = df[
        [
            "sequence",
            "split",
            "y_true_eff",
            "mu_pred",
            "risk_true_proxy",
            "risk_pred",
            "sigma_pred",
            "utility_true",
            "score_on_target_only",
            "score_off_target_only",
            "score_integrated",
        ]
    ].copy()
    ranked["rank_integrated"] = ranked["score_integrated"].rank(method="first", ascending=False).astype(int)
    ranked["rank_on_target_only"] = ranked["score_on_target_only"].rank(method="first", ascending=False).astype(int)
    ranked["rank_off_target_only"] = ranked["score_off_target_only"].rank(method="first", ascending=False).astype(int)
    ranked = ranked.sort_values(["split", "rank_integrated"], ascending=[True, True]).reset_index(drop=True)
    out_dir_csv = os.path.dirname(args.output_ranked_csv)
    if out_dir_csv:
        os.makedirs(out_dir_csv, exist_ok=True)
    ranked.to_csv(args.output_ranked_csv, index=False)

    print("Saved:", args.output_json)
    print("Saved:", args.output_ranked_csv)
    print("Test NDCG@20 | on_target_only=", f"{m_on.ndcg20:.6f}", "off_target_only=", f"{m_off.ndcg20:.6f}", "integrated=", f"{m_int.ndcg20:.6f}")
    print("Test Spearman | on_target_only=", f"{m_on.spearman:.6f}", "off_target_only=", f"{m_off.spearman:.6f}", "integrated=", f"{m_int.spearman:.6f}")


if __name__ == "__main__":
    main()
