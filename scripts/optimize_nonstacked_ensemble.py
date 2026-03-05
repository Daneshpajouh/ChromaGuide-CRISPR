#!/usr/bin/env python3
"""Optimize a split-specific non-stacked ensemble from run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SKIP_NAME_TOKENS = (
    "summary",
    "submitted",
    "significance",
    "conformal",
    "offtarget",
    "ensemble",
    "proposal_status",
    "runtime_snapshot",
)
STACKED_TOKENS = ("oof", "stack", "meta", "rankfusion")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize non-stacked ensemble")
    p.add_argument("--results_dir", type=Path, default=Path("results/runs"))
    p.add_argument("--split", choices=["A", "B", "C"], default="A")
    p.add_argument("--max_models", type=int, default=160)
    p.add_argument("--max_ensemble_size", type=int, default=32)
    p.add_argument("--random_trials", type=int, default=200000)
    p.add_argument("--local_trials", type=int, default=30000)
    p.add_argument("--local_sigma", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--seed_members_json",
        type=Path,
        default=None,
        help="Optional JSON file with a 'members' list used as seeded ensemble set",
    )
    p.add_argument(
        "--output_json",
        type=Path,
        default=Path("results/runs/nonstacked_superopt_summary.json"),
    )
    p.add_argument(
        "--output_pred_csv",
        type=Path,
        default=Path("results/runs/nonstacked_superopt_predictions.csv"),
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
            candidates.extend([Path.cwd() / p, results_dir.parent / p, results_dir / p.name])
        for cand in candidates:
            if cand.exists():
                return cand
    p2 = metrics_path.with_name(metrics_path.stem + "_predictions.csv")
    return p2 if p2.exists() else None


def looks_like_split(payload: dict[str, Any], metrics_name: str, split: str) -> bool:
    split_field = str(payload.get("split", "")).upper()
    if split_field == split:
        return True
    return f"split{split.lower()}" in metrics_name.lower()


def is_candidate(metrics_name: str) -> bool:
    lname = metrics_name.lower()
    if any(t in lname for t in SKIP_NAME_TOKENS):
        return False
    if any(t in lname for t in STACKED_TOKENS):
        return False
    return True


def load_models(results_dir: Path, split: str) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for metrics_path in sorted(results_dir.glob("*.json")):
        if not is_candidate(metrics_path.name):
            continue
        try:
            payload = json.loads(metrics_path.read_text())
        except Exception:
            continue
        if not isinstance(payload, dict) or "gold_rho" not in payload:
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
        if not {"efficiency", "prediction"}.issubset(pred_df.columns):
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
        rho_csv = maybe_float(spearmanr(pred_df["efficiency"].values, pred_df["prediction"].values)[0], -1.0)
        models.append(
            {
                "name": metrics_path.stem,
                "metrics_file": str(metrics_path),
                "predictions_file": str(pred_path),
                "rho_json": maybe_float(payload.get("gold_rho"), -1.0),
                "rho": rho_csv,
                "n_rows": int(len(pred_df)),
                "prediction": pred_df["prediction"].to_numpy(dtype=np.float64),
                "efficiency": pred_df["efficiency"].to_numpy(dtype=np.float64),
                "sequence": seq_values,
                "row_key": row_key,
            }
        )

    if not models:
        raise RuntimeError(f"No usable non-stacked split-{split} model outputs found in {results_dir}")

    n0 = models[0]["n_rows"]
    y0 = models[0]["efficiency"]
    ref_keys = models[0]["row_key"]
    ref_index = {k: i for i, k in enumerate(ref_keys.tolist())}
    filtered: list[dict[str, Any]] = []
    for m in models:
        if m["n_rows"] != n0 or len(m["row_key"]) != len(ref_keys):
            continue
        if np.array_equal(m["row_key"], ref_keys):
            if np.max(np.abs(m["efficiency"] - y0)) > 1e-6:
                continue
            filtered.append(m)
            continue
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
        raise RuntimeError("Found non-stacked models, but none aligned to shared evaluation rows")
    filtered.sort(key=lambda x: x["rho"], reverse=True)
    return filtered


def greedy_select(models: list[dict[str, Any]], max_size: int) -> tuple[list[int], np.ndarray, float]:
    y = models[0]["efficiency"]
    preds = [m["prediction"] for m in models]
    selected = [0]
    ens = preds[0].copy()
    best = maybe_float(spearmanr(y, ens)[0], -1.0)
    while len(selected) < min(max_size, len(models)):
        best_try = None
        for idx in range(len(models)):
            if idx in selected:
                continue
            cand = (ens * len(selected) + preds[idx]) / (len(selected) + 1)
            rho = maybe_float(spearmanr(y, cand)[0], -1.0)
            if best_try is None or rho > best_try[0]:
                best_try = (rho, idx, cand)
        if best_try is None or best_try[0] <= best:
            break
        best, idx, ens = best_try
        selected.append(idx)
    return selected, ens, best


def seeded_select(
    models: list[dict[str, Any]], seed_members_json: Path, max_size: int
) -> tuple[list[int], np.ndarray | None]:
    name_to_idx = {m["name"]: i for i, m in enumerate(models)}
    payload = json.loads(seed_members_json.read_text())
    members = payload.get("members", [])
    weights = payload.get("weights", [])
    selected = []
    seed_weights = []
    for name in members:
        if name in name_to_idx and name_to_idx[name] not in selected:
            selected.append(name_to_idx[name])
            try:
                i = members.index(name)
                seed_weights.append(float(weights[i]))
            except Exception:
                seed_weights.append(0.0)
    if not selected:
        return selected, None
    for idx in range(len(models)):
        if len(selected) >= max_size:
            break
        if idx not in selected:
            selected.append(idx)
            seed_weights.append(0.0)
    w = np.asarray(seed_weights, dtype=np.float64)
    if len(w) != len(selected):
        return selected, None
    sw = np.sum(w)
    if sw <= 0:
        return selected, None
    return selected, (w / sw)


def optimize_weights(
    models: list[dict[str, Any]],
    selected: list[int],
    random_trials: int,
    local_trials: int,
    local_sigma: float,
    seed: int,
    init_w: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    y = models[0]["efficiency"]
    stack = np.stack([models[i]["prediction"] for i in selected], axis=0)
    rng = np.random.default_rng(seed)
    n = len(selected)

    best_w = np.ones(n, dtype=np.float64) / n
    best_pred = np.tensordot(best_w, stack, axes=(0, 0))
    best_rho = maybe_float(spearmanr(y, best_pred)[0], -1.0)
    if init_w is not None and len(init_w) == n:
        iw = np.asarray(init_w, dtype=np.float64)
        sw = np.sum(iw)
        if sw > 0:
            iw = iw / sw
            ip = np.tensordot(iw, stack, axes=(0, 0))
            ir = maybe_float(spearmanr(y, ip)[0], -1.0)
            if ir > best_rho:
                best_rho = ir
                best_w = iw
                best_pred = ip

    for _ in range(max(0, random_trials)):
        w = rng.dirichlet(np.ones(n, dtype=np.float64))
        cand = np.tensordot(w, stack, axes=(0, 0))
        rho = maybe_float(spearmanr(y, cand)[0], -1.0)
        if rho > best_rho:
            best_rho = rho
            best_w = w
            best_pred = cand

    for _ in range(max(0, local_trials)):
        noise = rng.normal(0.0, local_sigma, size=n)
        w = best_w * np.exp(noise)
        s = np.sum(w)
        if s <= 0:
            continue
        w = w / s
        cand = np.tensordot(w, stack, axes=(0, 0))
        rho = maybe_float(spearmanr(y, cand)[0], -1.0)
        if rho > best_rho:
            best_rho = rho
            best_w = w
            best_pred = cand

    return best_pred, best_rho, best_w


def main() -> None:
    args = parse_args()
    models = load_models(args.results_dir, args.split)
    top = models[: max(1, args.max_models)]
    selected: list[int]
    init_w: np.ndarray | None = None
    if args.seed_members_json is not None and args.seed_members_json.exists():
        selected, init_w = seeded_select(top, args.seed_members_json, args.max_ensemble_size)
        if not selected:
            selected, _, _ = greedy_select(top, args.max_ensemble_size)
    else:
        selected, _, _ = greedy_select(top, args.max_ensemble_size)
    ens_pred, ens_rho, weights = optimize_weights(
        top,
        selected,
        random_trials=args.random_trials,
        local_trials=args.local_trials,
        local_sigma=args.local_sigma,
        seed=args.seed,
        init_w=init_w,
    )

    seq = top[0]["sequence"]
    y = top[0]["efficiency"]
    out_df = pd.DataFrame({"sequence": seq, "efficiency": y, "prediction": ens_pred})
    args.output_pred_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_pred_csv, index=False)

    payload = {
        "scope": "nonstacked_only",
        "split": args.split,
        "best_rho": ens_rho,
        "selected_size": len(selected),
        "best_single_model": {"name": top[0]["name"], "rho": top[0]["rho"]},
        "members": [top[i]["name"] for i in selected],
        "weights": [float(x) for x in weights.tolist()],
        "predictions_file": str(args.output_pred_csv),
        "n_models_considered": len(top),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))

    print(f"Best nonstacked single: {top[0]['name']} rho={top[0]['rho']:.6f}")
    print(f"Best nonstacked ensemble ({len(selected)}): rho={ens_rho:.6f}")
    print(f"Wrote: {args.output_json}")
    print(f"Wrote: {args.output_pred_csv}")


if __name__ == "__main__":
    main()
