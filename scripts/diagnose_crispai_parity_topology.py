#!/usr/bin/env python3
"""Diagnose crispAI parity divergence on the exact public bundle.

This script does not retrain. It compares multiple evaluation topologies against the
exact staged test CSV, checkpoint outputs, and the shipped Fig4 arrays to isolate
whether the remaining gap to the frozen 0.5114 target is due to:
- target column choice
- transform choice
- aggregation choice
- shipped-array mismatch
- dropped-row handling
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import PowerTransformer
from torch.utils.data import DataLoader, TensorDataset


CANDIDATE_TARGET_COLS = [
    "CHANGEseq_reads_adjusted",
    "CHANGEseq_reads",
    "Active",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument(
        "--source-dir",
        default="data/public_benchmarks/off_target/crispai_parity/model_bundle",
    )
    ap.add_argument(
        "--parity-dir",
        default="data/public_benchmarks/off_target/crispai_parity",
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument(
        "--output-json",
        default="results/public_benchmarks/crispai_parity_topology_diagnosis.json",
    )
    return ap.parse_args()



def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    n = min(len(a), len(b))
    return float(spearmanr(a[:n], b[:n]).correlation)



def boxcox_score(arr: np.ndarray, eps: float, jitter: bool, seed: int = 0) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1, 1)
    if jitter:
        rng = np.random.default_rng(seed)
        arr = arr + rng.normal(1, 1e-6, len(arr)).reshape(-1, 1)
    pt = PowerTransformer(method="box-cox", standardize=True)
    return pt.fit_transform(arr + eps).reshape(-1)



def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    source_dir = (repo_root / args.source_dir).resolve()
    parity_dir = (repo_root / args.parity_dir).resolve()
    out_json = (repo_root / args.output_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(source_dir))
    from model import CrispAI_pi  # type: ignore
    from utils import preprocess_features  # type: ignore

    test_csv = parity_dir / "data_bundle" / "changeseq_offtarget_data_flank73_filtered_nupop_gc_bdm_preprocessed_test.csv"
    checkpoint_path = parity_dir / "data_bundle" / "epoch:19-best_valid_loss:0.270.pt"
    shipped_preds_path = parity_dir / "data_bundle" / "changeseqtest_preds_median_scores_all.npy"
    shipped_y_path = parity_dir / "data_bundle" / "y_test.npy"

    df_raw = pd.read_csv(test_csv)
    raw_rows = int(len(df_raw))
    raw_columns = list(df_raw.columns)

    df = preprocess_features(
        df=df_raw.copy(),
        reads="CHANGEseq_reads",
        target="target",
        offtarget_sequence="offtarget_sequence",
        distance="distance",
        read_cutoff=10,
        max_reads=1e4,
        nupop_occupancy_col="NuPoP occupancy",
        nupop_affinity_col="NuPoP affinity",
        gc_content_col="GC flank73",
        nucleotide_bdm_col="nucleotide BDM",
    )
    processed_rows = int(len(df))

    x = np.stack([row.astype(np.float32) for row in df["interface_encoding"].values], axis=0)
    x_pi = np.stack([row.astype(np.float32) for row in df["physical_features"].values], axis=0)
    x = np.concatenate([x, x_pi], axis=2)

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CrispAI_pi(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    preds = []
    loader = DataLoader(TensorDataset(torch.tensor(x)), batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            samples = model.draw_samples(xb, n_samples=args.n_samples).T
            preds.append(samples)
    preds = np.concatenate(preds, axis=0)
    preds_mean = np.mean(preds, axis=1)
    preds_median = np.median(preds, axis=1)

    shipped_preds = np.load(shipped_preds_path)
    shipped_y = np.load(shipped_y_path)

    diagnostics: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_csv": str(test_csv),
        "checkpoint_path": str(checkpoint_path),
        "raw_rows": raw_rows,
        "processed_rows": processed_rows,
        "dropped_rows": raw_rows - processed_rows,
        "raw_columns": raw_columns,
        "candidate_target_cols_present": [c for c in CANDIDATE_TARGET_COLS if c in df.columns],
        "shipped_arrays": {
            "preds_shape": list(shipped_preds.shape),
            "y_shape": list(shipped_y.shape),
            "raw_spearman_shipped_arrays": safe_spearman(shipped_preds, shipped_y),
        },
        "local_vs_shipped": {
            "preds_median_raw_vs_shipped_preds_spearman": safe_spearman(preds_median, shipped_preds),
            "preds_mean_raw_vs_shipped_preds_spearman": safe_spearman(preds_mean, shipped_preds),
        },
        "target_sweeps": [],
    }

    for col in [c for c in CANDIDATE_TARGET_COLS if c in df.columns]:
        y = np.asarray(df[col].values, dtype=float)
        entry = {
            "target_col": col,
            "raw_mean_spearman": safe_spearman(preds_mean, y),
            "raw_median_spearman": safe_spearman(preds_median, y),
            "boxcox_mean_with_jitter_spearman": safe_spearman(
                boxcox_score(preds_mean, model.eps, jitter=True),
                boxcox_score(y, model.eps, jitter=False),
            ),
            "boxcox_median_no_jitter_spearman": safe_spearman(
                boxcox_score(preds_median, model.eps, jitter=False),
                boxcox_score(y, model.eps, jitter=False),
            ),
            "boxcox_mean_no_jitter_spearman": safe_spearman(
                boxcox_score(preds_mean, model.eps, jitter=False),
                boxcox_score(y, model.eps, jitter=False),
            ),
            "shipped_y_raw_spearman_against_target": safe_spearman(shipped_y, y),
        }
        diagnostics["target_sweeps"].append(entry)

    diagnostics["best_boxcox_median_no_jitter"] = max(
        diagnostics["target_sweeps"], key=lambda x: x["boxcox_median_no_jitter_spearman"]
    )
    diagnostics["best_raw_median"] = max(
        diagnostics["target_sweeps"], key=lambda x: x["raw_median_spearman"]
    )
    diagnostics["frozen_target"] = 0.5114
    diagnostics["best_boxcox_median_no_jitter_gap"] = (
        diagnostics["best_boxcox_median_no_jitter"]["boxcox_median_no_jitter_spearman"] - 0.5114
    )

    out_json.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
