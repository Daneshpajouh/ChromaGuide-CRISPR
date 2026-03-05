#!/usr/bin/env python3
"""Run matched public on-target transfer evaluation using staged fold directories."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description="Run public on-target transfer evaluation.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--source-dataset", required=True)
    ap.add_argument("--target-dataset", required=True)
    ap.add_argument("--folds", type=int, default=5, help="Run same-index source/target fold pairs from 0..folds-1.")
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
        default="results/public_benchmarks/transfer_runs",
        help="Where per-run outputs should be written.",
    )
    ap.add_argument(
        "--summary-json",
        default="results/public_benchmarks/transfer_runs/run_manifest.json",
        help="Write run manifest here.",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    output_root = repo / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for fold in range(args.folds):
        subprocess.run(
            [
                sys.executable,
                "scripts/build_public_on_target_transfer_split.py",
                "--repo-root",
                str(repo),
                "--source-dataset",
                args.source_dataset,
                "--target-dataset",
                args.target_dataset,
                "--source-fold",
                str(fold),
                "--target-fold",
                str(fold),
            ],
            check=True,
            cwd=repo,
            env=_clean_env(),
        )
        split_dir = (
            repo
            / "data"
            / "public_benchmarks"
            / "on_target"
            / "transfer"
            / f"{args.source_dataset}_to_{args.target_dataset}"
            / f"fold_{fold}_{fold}"
        )
        output_prefix = output_root / f"{args.source_dataset}_to_{args.target_dataset}_fold{fold}"
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
        manifest.append(
            {
                "source_dataset": args.source_dataset,
                "target_dataset": args.target_dataset,
                "fold_pair": [fold, fold],
                "split_dir": str(split_dir),
                "output_prefix": str(output_prefix),
                "command": cmd,
            }
        )
        subprocess.run(cmd, check=True, cwd=repo, env=_clean_env())

    summary_path = repo / args.summary_json
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"runs": manifest}, indent=2), encoding="utf-8")
    print(json.dumps({"runs": manifest}, indent=2))


if __name__ == "__main__":
    main()
