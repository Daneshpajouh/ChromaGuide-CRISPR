#!/usr/bin/env python3
"""Run a crispAI parity evaluation on the exact staged test frame.

This mirrors the published supplementary evaluation path as closely as possible
using the shipped test CSV, best checkpoint, and Box-Cox scoring topology.
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
from sklearn.preprocessing import PowerTransformer
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument(
        "--source-dir",
        default="data/public_benchmarks/sources/crispAI_crispr-offtarget-uncertainty/crispAI_score",
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
        default="results/public_benchmarks/crispai_parity_eval.json",
    )
    ap.add_argument(
        "--compare-shipped-fig4",
        action="store_true",
        help="Compare locally reproduced arrays against shipped Fig4 arrays when present.",
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


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    source_dir = (repo_root / args.source_dir).resolve()
    parity_dir = (repo_root / args.parity_dir).resolve()
    out_json = (repo_root / args.output_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"

    compat_script = repo_root / "scripts" / "apply_crispai_upstream_compat.py"
    os.system(f"{sys.executable} {compat_script} --repo-root {repo_root} >/dev/null")

    sys.path.insert(0, str(source_dir))
    from model import CrispAI_pi  # type: ignore
    from utils import preprocess_features  # type: ignore

    test_csv = parity_dir / "data_bundle" / "changeseq_offtarget_data_flank73_filtered_nupop_gc_bdm_preprocessed_test.csv"
    checkpoint_path = parity_dir / "data_bundle" / "epoch:19-best_valid_loss:0.270.pt"
    manifest_path = parity_dir / "crispai_parity_manifest.json"

    df_test = pd.read_csv(test_csv)
    df_test = preprocess_features(
        df=df_test,
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

    x = np.stack([row.astype(np.float32) for row in df_test["interface_encoding"].values], axis=0)
    x_pi = np.stack([row.astype(np.float32) for row in df_test["physical_features"].values], axis=0)
    x = np.concatenate([x, x_pi], axis=2)
    y = np.stack([row.astype(np.float32) for row in df_test["CHANGEseq_reads_adjusted"].values], axis=0)

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = CrispAI_pi(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    preds = []
    loader = DataLoader(TensorDataset(torch.tensor(x)), batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device)
            samples = model.draw_samples(xb, n_samples=args.n_samples).T
            preds.append(samples)

    preds = np.concatenate(preds, axis=0)
    preds_mean = np.mean(preds, axis=1)
    preds_median = np.median(preds, axis=1)

    pt = PowerTransformer(method="box-cox", standardize=True)
    rng = np.random.default_rng(0)
    preds_scores = pt.fit_transform(
        preds_mean.reshape(-1, 1) + model.eps + rng.normal(1, 1e-6, len(preds_mean)).reshape(-1, 1)
    ).reshape(-1)
    preds_median_scores = pt.fit_transform(
        preds_median.reshape(-1, 1) + model.eps
    ).reshape(-1)
    y_test = pt.fit_transform(y.reshape(-1, 1) + model.eps).reshape(-1)

    preds_spearman = float(pd.Series(preds_scores).corr(pd.Series(y_test), method="spearman"))
    preds_median_spearman = float(pd.Series(preds_median_scores).corr(pd.Series(y_test), method="spearman"))

    fig4 = {}
    if args.compare_shipped_fig4:
        fig4_preds_path = parity_dir / "data_bundle" / "changeseqtest_preds_median_scores_all.npy"
        fig4_y_path = parity_dir / "data_bundle" / "y_test.npy"
        if fig4_preds_path.exists() and fig4_y_path.exists():
            shipped_preds = np.load(fig4_preds_path)
            shipped_y = np.load(fig4_y_path)
            fig4["available"] = True
            fig4["preds_shape"] = list(shipped_preds.shape)
            fig4["y_shape"] = list(shipped_y.shape)
            fig4["raw_spearman_shipped_arrays"] = float(
                pd.Series(shipped_preds).corr(pd.Series(shipped_y), method="spearman")
            )
            fig4["raw_spearman_local_median_vs_shipped_y"] = float(
                pd.Series(preds_median).corr(pd.Series(shipped_y[: len(preds_median)]), method="spearman")
            )
            fig4["transformed_spearman_local_median_vs_local_y"] = preds_median_spearman
            fig4["median_score_mse_vs_shipped"] = float(
                np.mean((preds_median_scores - shipped_preds[: len(preds_median_scores)]) ** 2)
            )
            fig4["y_test_mse_vs_shipped"] = float(
                np.mean((y_test - shipped_y[: len(y_test)]) ** 2)
            )
        else:
            fig4["available"] = False

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "source_dir": str(source_dir),
        "parity_dir": str(parity_dir),
        "parity_manifest": str(manifest_path),
        "test_csv": str(test_csv),
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "batch_size": args.batch_size,
        "n_samples": args.n_samples,
        "rows": int(len(df_test)),
        "metrics": {
            "preds_mean_spearman": preds_spearman,
            "preds_median_spearman": preds_median_spearman,
        },
        "array_diagnostics": {
            "preds_mean_min": float(np.min(preds_mean)),
            "preds_mean_max": float(np.max(preds_mean)),
            "preds_median_min": float(np.min(preds_median)),
            "preds_median_max": float(np.max(preds_median)),
            "y_min": float(np.min(y)),
            "y_max": float(np.max(y)),
        },
        "claim_target": {
            "CHANGE_seq_test_Spearman": 0.5114,
            "gap_mean_vs_target": preds_spearman - 0.5114,
            "gap_median_vs_target": preds_median_spearman - 0.5114,
        },
        "fig4_comparison": fig4,
        "notes": [
            "Runs the exact staged crispAI test CSV and best checkpoint.",
            "Mean path uses the supplementary Box-Cox + jitter topology.",
            "Median path mirrors the supplementary script and does not inject jitter.",
        ],
        "status": "ok",
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
