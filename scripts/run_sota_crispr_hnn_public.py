#!/usr/bin/env python3
"""Run CRISPR_HNN architecture on frozen public on-target folds.

This reconstructs the upstream CRISPR_HNN model from the cloned source repo but
uses our deterministic public fold artifacts for matched comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CANONICAL_9 = [
    "WT",
    "ESP",
    "HF",
    "xCas9",
    "SpCas9-NG",
    "Sniper-Cas9",
    "HCT116",
    "HELA",
    "HL60",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run reconstructed CRISPR_HNN on public folds.")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--datasets", default=",".join(CANONICAL_9))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--output-root", default="results/public_benchmarks/sota_crispr_hnn")
    p.add_argument("--summary-json", default="")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def seq_to_onehot_and_token(seq: str) -> tuple[np.ndarray, np.ndarray]:
    code = {"A": [1, 0, 0, 0], "C": [0, 1, 0, 0], "G": [0, 0, 1, 0], "T": [0, 0, 0, 1]}
    tok = {"A": 2, "C": 3, "G": 4, "T": 5}
    s = (seq or "").upper()[:23]
    if len(s) < 23:
        s = s + "N" * (23 - len(s))

    onehot = np.zeros((1, 23, 4), dtype=np.float32)
    token = np.ones((24,), dtype=np.int32)
    for i, ch in enumerate(s):
        if ch in code:
            onehot[0, i] = np.array(code[ch], dtype=np.float32)
            token[i + 1] = tok[ch]
    return onehot, token


def read_rows(path: Path, max_rows: int = 0) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["sequence"].strip(), float(row["efficiency"])))
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def dataset_split_paths(repo_root: Path, dataset: str, fold_idx: int) -> tuple[Path, Path, Path]:
    base = repo_root / "data" / "public_benchmarks" / "on_target" / "folds" / dataset / f"fold_{fold_idx}"
    tr = base / f"{dataset}_train.csv"
    va = base / f"{dataset}_validation.csv"
    te = base / f"{dataset}_test.csv"
    return tr, va, te


def build_arrays(rows: list[tuple[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1, x2, y = [], [], []
    for seq, val in rows:
        oh, tk = seq_to_onehot_and_token(seq)
        x1.append(oh)
        x2.append(tk)
        y.append(val)
    x1_arr = np.array(x1, dtype=np.float32).reshape((len(x1), 1, 23, 4))
    x2_arr = np.array(x2, dtype=np.int32)
    y_arr = np.array(y, dtype=np.float32)
    return x1_arr, x2_arr, y_arr


def corr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    import scipy as sp

    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    scc = float(sp.stats.spearmanr(y_true, y_pred)[0])
    pcc = float(sp.stats.pearsonr(y_true, y_pred)[0])
    return scc, pcc


def import_crispr_hnn(repo_root: Path):
    src = repo_root / "data" / "public_benchmarks" / "sources" / "CRISPR_HNN"
    if not src.exists():
        raise FileNotFoundError(f"CRISPR_HNN source not found: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from model import CRISPR_HNN  # type: ignore
    import tensorflow as tf  # type: ignore

    return CRISPR_HNN, tf


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_root = (repo_root / args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    CRISPR_HNN, tf = import_crispr_hnn(repo_root)
    tf.random.set_seed(args.seed)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    summary: dict[str, dict] = {}
    for dataset in datasets:
        fold_records: list[dict] = []
        for fold_idx in range(args.folds):
            train_csv, val_csv, test_csv = dataset_split_paths(repo_root, dataset, fold_idx)
            if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
                raise FileNotFoundError(f"Missing fold files for {dataset} fold {fold_idx}")

            train_rows = read_rows(train_csv, max_rows=args.max_rows)
            val_rows = read_rows(val_csv, max_rows=args.max_rows)
            test_rows = read_rows(test_csv, max_rows=args.max_rows)

            x1_tr, x2_tr, y_tr = build_arrays(train_rows)
            x1_val, x2_val, y_val = build_arrays(val_rows)
            x1_te, x2_te, y_te = build_arrays(test_rows)

            model = CRISPR_HNN()
            model.compile(
                loss="mean_absolute_error",
                optimizer=tf.keras.optimizers.Adamax(learning_rate=args.lr),
                metrics=["mae"],
            )
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=max(2, args.patience // 2), verbose=0),
            ]
            hist = model.fit(
                [x1_tr, x2_tr],
                y_tr,
                validation_data=([x1_val, x2_val], y_val),
                batch_size=args.batch_size,
                epochs=args.epochs,
                shuffle=False,
                verbose=0,
                callbacks=callbacks,
            )

            pred = model.predict([x1_te, x2_te], verbose=0).reshape(-1)
            scc, pcc = corr(y_te, pred)
            rec = {
                "dataset": dataset,
                "fold": fold_idx,
                "test_scc": scc,
                "test_pcc": pcc,
                "best_val_mae": float(min(hist.history.get("val_mae", [np.nan]))),
                "train_rows": int(len(train_rows)),
                "val_rows": int(len(val_rows)),
                "test_rows": int(len(test_rows)),
                "epochs_ran": int(len(hist.history.get("loss", []))),
            }
            fold_records.append(rec)
            (out_root / f"{dataset}_fold{fold_idx}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

        sccs = [r["test_scc"] for r in fold_records]
        pccs = [r["test_pcc"] for r in fold_records]
        summary[dataset] = {
            "dataset": dataset,
            "folds": fold_records,
            "mean_scc": float(np.mean(sccs)),
            "std_scc": float(np.std(sccs)),
            "mean_pcc": float(np.mean(pccs)),
            "std_pcc": float(np.std(pccs)),
        }

    means = [v["mean_scc"] for v in summary.values()]
    payload = {
        "model": "CRISPR_HNN_reconstructed_upstream_arch",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "datasets": datasets,
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patience": args.patience,
            "seed": args.seed,
            "max_rows": args.max_rows,
        },
        "dataset_summaries": summary,
        "mean_scc_across_datasets": float(np.mean(means)) if means else None,
    }
    summary_json = Path(args.summary_json).resolve() if args.summary_json else out_root / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
