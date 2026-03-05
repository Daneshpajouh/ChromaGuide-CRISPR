#!/usr/bin/env python3
"""Run all splits from an off-target manifest and aggregate metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--negative-keep-prob", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-channels", type=int, default=256)
    parser.add_argument("--fc-hidden", type=int, default=256)
    parser.add_argument("--conv-dropout", type=float, default=0.4)
    parser.add_argument("--fc-dropout", type=float, default=0.3)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest_path = Path(args.manifest_json)
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    payload = json.loads(manifest_path.read_text())
    splits = payload.get("splits", [])
    if not splits:
        raise RuntimeError(f"No splits found in manifest: {manifest_path}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, split in enumerate(splits):
        held_out = split.get("held_out_method", f"split_{idx}")
        split_slug = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(held_out))
        out_json = output_dir / f"split_{idx:02d}_{split_slug}.json"
        out_model = output_dir / f"split_{idx:02d}_{split_slug}.pt"
        cmd = [
            args.python_bin,
            "scripts/train_public_off_target_cclmoff.py",
            "--manifest-json",
            str(manifest_path),
            "--split-mode",
            "manifest",
            "--fold-index",
            str(idx),
            "--device",
            args.device,
            "--max_rows",
            str(args.max_rows),
            "--negative_keep_prob",
            str(args.negative_keep_prob),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight_decay",
            str(args.weight_decay),
            "--base_channels",
            str(args.base_channels),
            "--fc_hidden",
            str(args.fc_hidden),
            "--conv_dropout",
            str(args.conv_dropout),
            "--fc_dropout",
            str(args.fc_dropout),
            "--focal_alpha",
            str(args.focal_alpha),
            "--focal_gamma",
            str(args.focal_gamma),
            "--seed",
            str(args.seed),
            "--output_json",
            str(out_json),
            "--model_out",
            str(out_model),
        ]
        subprocess.run(cmd, cwd=repo_root, check=True)
        split_metrics = json.loads(out_json.read_text())
        rows.append(
            {
                "split_index": idx,
                "held_out_method": held_out,
                "best_auroc": split_metrics.get("best_auroc"),
                "best_auprc": split_metrics.get("best_auprc"),
                "n_train_rows": split_metrics.get("n_train_rows"),
                "n_val_rows": split_metrics.get("n_val_rows"),
                "metrics_json": str(out_json),
            }
        )

    valid_auc = [float(r["best_auroc"]) for r in rows if r.get("best_auroc") is not None]
    valid_pr = [float(r["best_auprc"]) for r in rows if r.get("best_auprc") is not None]
    summary = {
        "manifest_json": str(manifest_path),
        "frame_name": payload.get("frame_name"),
        "split_mode": payload.get("split_mode"),
        "n_splits": len(rows),
        "mean_auroc": (sum(valid_auc) / len(valid_auc)) if valid_auc else None,
        "mean_auprc": (sum(valid_pr) / len(valid_pr)) if valid_pr else None,
        "min_auroc": min(valid_auc) if valid_auc else None,
        "min_auprc": min(valid_pr) if valid_pr else None,
        "max_auroc": max(valid_auc) if valid_auc else None,
        "max_auprc": max(valid_pr) if valid_pr else None,
        "splits": rows,
    }
    out_summary = Path(args.summary_json)
    if not out_summary.is_absolute():
        out_summary = (repo_root / out_summary).resolve()
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2))
    print(str(out_summary))


if __name__ == "__main__":
    main()
