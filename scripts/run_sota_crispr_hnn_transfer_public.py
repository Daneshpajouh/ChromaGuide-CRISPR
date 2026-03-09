#!/usr/bin/env python3
"""Run reconstructed CRISPR_HNN on frozen public transfer splits."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run reconstructed CRISPR_HNN on public transfer splits.")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--source-dataset", required=True)
    p.add_argument("--target-dataset", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--output-root", default="results/public_benchmarks/sota_crispr_hnn_transfer")
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


def build_transfer_split(repo_root: Path, source_dataset: str, target_dataset: str, fold: int) -> Path:
    cmd = [
        sys.executable,
        "scripts/build_public_on_target_transfer_split.py",
        "--repo-root",
        str(repo_root),
        "--source-dataset",
        source_dataset,
        "--target-dataset",
        target_dataset,
        "--source-fold",
        str(fold),
        "--target-fold",
        str(fold),
    ]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)
    return (
        repo_root
        / "data"
        / "public_benchmarks"
        / "on_target"
        / "transfer"
        / f"{source_dataset}_to_{target_dataset}"
        / f"fold_{fold}_{fold}"
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_root = (repo_root / args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    CRISPR_HNN, tf = import_crispr_hnn(repo_root)
    tf.random.set_seed(args.seed)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    fold_records: list[dict] = []
    for fold_idx in range(args.folds):
        split_dir = build_transfer_split(repo_root, args.source_dataset, args.target_dataset, fold_idx)
        train_csv = split_dir / f"{args.source_dataset}_train.csv"
        val_csv = split_dir / f"{args.source_dataset}_validation.csv"
        test_csv = split_dir / f"{args.target_dataset}_test.csv"

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
            "source_dataset": args.source_dataset,
            "target_dataset": args.target_dataset,
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
        (out_root / f"{args.source_dataset}_to_{args.target_dataset}_fold{fold_idx}.json").write_text(
            json.dumps(rec, indent=2), encoding="utf-8"
        )

    sccs = [r["test_scc"] for r in fold_records]
    pccs = [r["test_pcc"] for r in fold_records]
    payload = {
        "model": "CRISPR_HNN_reconstructed_upstream_arch_transfer",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "source_dataset": args.source_dataset,
            "target_dataset": args.target_dataset,
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patience": args.patience,
            "seed": args.seed,
            "max_rows": args.max_rows,
        },
        "folds": fold_records,
        "mean_scc": float(np.mean(sccs)),
        "std_scc": float(np.std(sccs)),
        "mean_pcc": float(np.mean(pccs)),
        "std_pcc": float(np.std(pccs)),
    }
    summary_json = Path(args.summary_json).resolve() if args.summary_json else out_root / "SUMMARY.json"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
