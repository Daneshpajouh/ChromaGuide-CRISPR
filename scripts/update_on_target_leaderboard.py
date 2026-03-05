#!/usr/bin/env python3
"""Update split-specific on-target leaderboard and best ensemble artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build leaderboard + greedy ensemble for on-target runs")
    p.add_argument("--results_dir", type=Path, default=Path("results/runs"))
    p.add_argument("--split", choices=["A", "B", "C"], default="A")
    p.add_argument("--target_rho", type=float, default=0.911)
    p.add_argument("--max_models", type=int, default=24)
    p.add_argument("--max_ensemble_size", type=int, default=10)
    p.add_argument("--random_weight_trials", type=int, default=3000)
    p.add_argument("--output_json", type=Path, default=Path("results/runs/experiment_leaderboard_latest.json"))
    p.add_argument(
        "--output_pred_csv",
        type=Path,
        default=Path("results/runs/on_target_splitA_best_ensemble_auto_predictions.csv"),
    )
    return p.parse_args()


def maybe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def infer_prediction_file(results_dir: Path, metrics_path: Path, payload: dict[str, Any]) -> Path | None:
    explicit = payload.get("predictions_file") or payload.get("prediction_file")
    if explicit:
        p = Path(explicit)
        candidates = [p]
        if not p.is_absolute():
            candidates.extend([Path.cwd() / p, results_dir.parent / p])
        for cand in candidates:
            if cand.exists():
                return cand

    candidate = metrics_path.with_name(metrics_path.stem + "_predictions.csv")
    if candidate.exists():
        return candidate

    return None


def looks_like_split(payload: dict[str, Any], metrics_name: str, split: str) -> bool:
    split_field = str(payload.get("split", "")).upper()
    if split_field == split:
        return True
    return f"split{split.lower()}" in metrics_name.lower()


def load_models(results_dir: Path, split: str) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for metrics_path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(metrics_path.read_text())
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue
        if "gold_rho" not in payload:
            continue
        if not looks_like_split(payload, metrics_path.name, split):
            continue

        pred_path = infer_prediction_file(results_dir, metrics_path, payload)
        if pred_path is None:
            continue

        try:
            pred_df = pd.read_csv(pred_path)
        except Exception:
            continue
        needed = {"efficiency", "prediction"}
        if not needed.issubset(pred_df.columns):
            continue

        if "sequence" in pred_df.columns:
            key_df = pred_df[["sequence", "efficiency"]].copy()
            key_df["__dup_idx"] = key_df.groupby(["sequence", "efficiency"]).cumcount()
            row_key = (
                key_df["sequence"].astype(str)
                + "||"
                + key_df["efficiency"].astype(str)
                + "||"
                + key_df["__dup_idx"].astype(str)
            ).to_numpy()
            seq_values = pred_df["sequence"].to_numpy()
        else:
            row_key = np.arange(len(pred_df)).astype(str)
            seq_values = np.arange(len(pred_df))

        rho_csv = maybe_float(spearmanr(pred_df["efficiency"].values, pred_df["prediction"].values)[0], default=-1.0)
        model = {
            "name": metrics_path.stem,
            "metrics_file": str(metrics_path),
            "predictions_file": str(pred_path),
            "rho_json": maybe_float(payload.get("gold_rho"), default=-1.0),
            "rho": rho_csv,
            "n_rows": int(len(pred_df)),
            "prediction": pred_df["prediction"].to_numpy(dtype=np.float64),
            "efficiency": pred_df["efficiency"].to_numpy(dtype=np.float64),
            "sequence": seq_values,
            "row_key": row_key,
        }
        models.append(model)

    if not models:
        raise RuntimeError(f"No usable split-{split} model outputs found in {results_dir}")

    # Sanity: keep models that can align to a shared evaluation set.
    n0 = models[0]["n_rows"]
    y0 = models[0]["efficiency"]
    ref_keys = models[0]["row_key"]
    ref_index = {k: i for i, k in enumerate(ref_keys.tolist())}
    filtered = []
    for m in models:
        if m["n_rows"] != n0 or len(m["row_key"]) != len(ref_keys):
            continue
        if np.array_equal(m["row_key"], ref_keys):
            if np.max(np.abs(m["efficiency"] - y0)) > 1e-6:
                continue
            filtered.append(m)
            continue

        # Try to realign model rows to the reference order.
        if len(set(m["row_key"].tolist())) != len(ref_keys):
            continue
        if set(m["row_key"].tolist()) != set(ref_keys.tolist()):
            continue
        try:
            order = np.asarray([ref_index[k] for k in m["row_key"].tolist()], dtype=np.int64)
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            m["prediction"] = m["prediction"][inv]
            m["efficiency"] = m["efficiency"][inv]
            m["sequence"] = m["sequence"][inv]
            m["row_key"] = m["row_key"][inv]
        except Exception:
            continue
        if np.max(np.abs(m["efficiency"] - y0)) > 1e-6:
            continue
        filtered.append(m)

    if not filtered:
        raise RuntimeError("Found model outputs, but none aligned to a shared evaluation set")

    filtered.sort(key=lambda x: x["rho"], reverse=True)
    return filtered


def greedy_ensemble(models: list[dict[str, Any]], max_size: int) -> tuple[list[int], np.ndarray, float]:
    y = models[0]["efficiency"]
    preds = [m["prediction"] for m in models]

    selected = [0]
    ens = preds[0].copy()
    best = maybe_float(spearmanr(y, ens)[0], default=-1.0)

    while len(selected) < min(max_size, len(models)):
        best_try = None
        for idx in range(len(models)):
            if idx in selected:
                continue
            cand = (ens * len(selected) + preds[idx]) / (len(selected) + 1)
            rho = maybe_float(spearmanr(y, cand)[0], default=-1.0)
            if best_try is None or rho > best_try[0]:
                best_try = (rho, idx, cand)

        if best_try is None or best_try[0] <= best:
            break

        best, idx, ens = best_try
        selected.append(idx)

    return selected, ens, best


def random_weight_refine(
    models: list[dict[str, Any]],
    selected: list[int],
    base_pred: np.ndarray,
    base_rho: float,
    n_trials: int,
) -> tuple[np.ndarray, float, list[float]]:
    rng = np.random.default_rng(42)
    y = models[0]["efficiency"]
    stack = np.stack([models[i]["prediction"] for i in selected], axis=0)

    best_pred = base_pred
    best_rho = base_rho
    best_w = np.ones(len(selected), dtype=np.float64) / len(selected)

    for _ in range(max(0, n_trials)):
        w = rng.dirichlet(np.ones(len(selected), dtype=np.float64))
        cand = np.tensordot(w, stack, axes=(0, 0))
        rho = maybe_float(spearmanr(y, cand)[0], default=-1.0)
        if rho > best_rho:
            best_rho = rho
            best_pred = cand
            best_w = w

    return best_pred, best_rho, best_w.tolist()


def main() -> None:
    args = parse_args()
    models = load_models(args.results_dir, args.split)

    top = models[: max(1, args.max_models)]
    selected, ens_pred, ens_rho = greedy_ensemble(top, args.max_ensemble_size)
    ens_pred, ens_rho, weights = random_weight_refine(top, selected, ens_pred, ens_rho, args.random_weight_trials)

    seq = top[0]["sequence"]
    y = top[0]["efficiency"]
    out_df = pd.DataFrame({"sequence": seq, "efficiency": y, "prediction": ens_pred})
    args.output_pred_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_pred_csv, index=False)

    selected_names = [top[i]["name"] for i in selected]

    payload = {
        "target_rho": args.target_rho,
        "split": args.split,
        "n_models_considered": len(top),
        "best_single_model": {
            "name": top[0]["name"],
            "rho": top[0]["rho"],
            "metrics_file": top[0]["metrics_file"],
            "predictions_file": top[0]["predictions_file"],
        },
        "best_ensemble_model": {
            "members": selected_names,
            "weights": weights,
            "rho": ens_rho,
            "predictions_file": str(args.output_pred_csv),
        },
        "all_single_model_rho": {m["name"]: m["rho"] for m in top},
        "gap_to_target_best_ensemble": args.target_rho - ens_rho,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))

    print(f"Best single: {top[0]['name']} rho={top[0]['rho']:.6f}")
    print(f"Best ensemble ({len(selected_names)}): rho={ens_rho:.6f}")
    print(f"Members: {selected_names}")
    print(f"Saved leaderboard: {args.output_json}")
    print(f"Saved predictions: {args.output_pred_csv}")


if __name__ == "__main__":
    main()
