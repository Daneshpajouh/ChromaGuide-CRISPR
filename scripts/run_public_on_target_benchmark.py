#!/usr/bin/env python3
"""Run the staged public on-target benchmark through the existing trainer.

This wrapper assumes public fold directories were already built.
Run this with the benchmark venv's Python interpreter for full ML dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


CANONICAL = [
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


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the staged public on-target benchmark through the existing trainer.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--datasets", nargs="*", default=["ALL"], help="Datasets to run, or ALL")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--encoder-type", default="cnn_gru")
    ap.add_argument("--fusion", default="gate")
    ap.add_argument("--loss-type", default="mse")
    ap.add_argument("--pretrain", choices=["none", "hfesp"], default="none")
    ap.add_argument("--pretrain-epochs", type=int, default=6)
    ap.add_argument("--pretrain-data-dir", default="")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--finetune-pretrain-mix", type=float, default=0.0)
    ap.add_argument("--finetune-pretrain-max-rows", type=int, default=0)
    ap.add_argument(
        "--output-root",
        default="results/public_benchmarks/full_runs",
        help="Where per-run outputs should be written.",
    )
    ap.add_argument(
        "--summary-json",
        default="results/public_benchmarks/full_runs/run_manifest.json",
        help="Write run manifest here.",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    datasets = CANONICAL if args.datasets == ["ALL"] else args.datasets
    output_root = repo / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    run_manifest = []
    for dataset in datasets:
        for fold in range(args.folds):
            split_dir = repo / "data" / "public_benchmarks" / "on_target" / "folds" / dataset / f"fold_{fold}"
            if not split_dir.exists():
                raise FileNotFoundError(split_dir)
            output_prefix = output_root / f"{dataset}_fold{fold}"
            cmd = [
                sys.executable,
                "scripts/train_on_target_trainval.py",
                "--split", "A",
                "--split_dir", str(split_dir),
                "--device", args.device,
                "--encoder_type", args.encoder_type,
                "--d_model", str(args.d_model),
                "--fusion", args.fusion,
                "--loss_type", args.loss_type,
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--dropout", str(args.dropout),
                "--finetune_epochs", str(args.epochs),
                "--patience", "3",
                "--pretrain", args.pretrain,
                "--pretrain_epochs", str(args.pretrain_epochs),
                "--finetune_pretrain_mix", str(args.finetune_pretrain_mix),
                "--finetune_pretrain_max_rows", str(args.finetune_pretrain_max_rows),
                "--use_explicit_validation_holdout",
                "--output_prefix", str(output_prefix),
            ]
            if args.pretrain_data_dir:
                cmd.extend(["--pretrain_data_dir", args.pretrain_data_dir])
            run_manifest.append({
                "dataset": dataset,
                "fold": fold,
                "split_dir": str(split_dir),
                "output_prefix": str(output_prefix),
                "command": cmd,
            })
            subprocess.run(cmd, check=True, cwd=repo, env=_clean_env())

    summary_path = repo / args.summary_json
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"runs": run_manifest}, indent=2), encoding="utf-8")
    print(json.dumps({"runs": run_manifest}, indent=2))


if __name__ == "__main__":
    main()
