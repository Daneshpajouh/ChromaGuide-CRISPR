#!/usr/bin/env python3
"""Build an out-of-fold meta-stacker on top of split-specific model predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from update_on_target_leaderboard import load_models


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-fit OOF meta-stacker for on-target predictions")
    p.add_argument("--results_dir", type=Path, default=Path("results/runs"))
    p.add_argument("--split", choices=["A", "B", "C"], default="A")
    p.add_argument("--top_k", type=int, default=300)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feature_type", choices=["rank", "raw"], default="rank")
    p.add_argument("--meta_model", choices=["ridge", "hgbr"], default="ridge")
    p.add_argument("--ridge_alpha", type=float, default=0.001)
    p.add_argument("--hgbr_depth", type=int, default=8)
    p.add_argument("--hgbr_iters", type=int, default=300)
    p.add_argument("--hgbr_lr", type=float, default=0.05)
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--output_pred_csv", type=Path, required=True)
    return p.parse_args()


def zscore_columns(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-12
    return (x - mu) / sd


def build_features(models: list[dict], top_k: int, feature_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    k = min(top_k, len(models))
    names = [m["name"] for m in models[:k]]
    y = models[0]["efficiency"].astype(np.float64)
    seq = models[0]["sequence"]

    cols = []
    for m in models[:k]:
        p = m["prediction"].astype(np.float64)
        if feature_type == "rank":
            p = rankdata(p).astype(np.float64)
        cols.append(p)
    x = np.stack(cols, axis=1)
    x = zscore_columns(x)
    return x, y, seq, names


def fit_predict_fold(args: argparse.Namespace, x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray) -> np.ndarray:
    if args.meta_model == "ridge":
        model = Ridge(alpha=args.ridge_alpha, random_state=args.seed)
    else:
        model = HistGradientBoostingRegressor(
            max_depth=args.hgbr_depth,
            max_iter=args.hgbr_iters,
            learning_rate=args.hgbr_lr,
            random_state=args.seed,
        )
    model.fit(x_tr, y_tr)
    return model.predict(x_va)


def main() -> None:
    args = parse_args()
    models = load_models(args.results_dir, args.split)
    x, y, seq, names = build_features(models, args.top_k, args.feature_type)

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(y), dtype=np.float64)

    for tr, va in kf.split(x):
        oof[va] = fit_predict_fold(args, x[tr], y[tr], x[va])

    rho = float(spearmanr(y, oof)[0])

    args.output_pred_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sequence": seq, "efficiency": y, "prediction": oof}).to_csv(args.output_pred_csv, index=False)

    payload = {
        "split": args.split,
        "gold_rho": rho,
        "encoder_type": f"oof_meta_{args.meta_model}",
        "fusion": "meta",
        "n_models": len(names),
        "feature_type": args.feature_type,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "meta_model": args.meta_model,
        "ridge_alpha": args.ridge_alpha,
        "hgbr_depth": args.hgbr_depth,
        "hgbr_iters": args.hgbr_iters,
        "hgbr_lr": args.hgbr_lr,
        "predictions_file": str(args.output_pred_csv.resolve()),
        "members": names,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))

    print(f"OOF rho: {rho:.6f}")
    print(f"Saved metrics: {args.output_json}")
    print(f"Saved predictions: {args.output_pred_csv}")


if __name__ == "__main__":
    main()
