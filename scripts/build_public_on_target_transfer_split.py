#!/usr/bin/env python3
"""Build a trainer-compatible transfer split from existing public fold directories."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PARTITIONS = ["train", "validation", "test"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a trainer-compatible transfer split from existing public fold directories.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--source-dataset", required=True)
    ap.add_argument("--target-dataset", required=True)
    ap.add_argument("--source-fold", type=int, default=0)
    ap.add_argument("--target-fold", type=int, default=0)
    ap.add_argument(
        "--output-dir",
        default="",
        help="Optional explicit output directory; defaults under data/public_benchmarks/on_target/transfer.",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    folds_root = repo / "data" / "public_benchmarks" / "on_target" / "folds"
    source_dir = folds_root / args.source_dataset / f"fold_{args.source_fold}"
    target_dir = folds_root / args.target_dataset / f"fold_{args.target_fold}"
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if not target_dir.exists():
        raise FileNotFoundError(target_dir)

    if args.output_dir:
        out_dir = repo / args.output_dir
    else:
        out_dir = repo / "data" / "public_benchmarks" / "on_target" / "transfer" / f"{args.source_dataset}_to_{args.target_dataset}" / f"fold_{args.source_fold}_{args.target_fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_dir / f"{args.source_dataset}_train.csv", out_dir / f"{args.source_dataset}_train.csv")
    shutil.copy2(source_dir / f"{args.source_dataset}_validation.csv", out_dir / f"{args.source_dataset}_validation.csv")
    shutil.copy2(target_dir / f"{args.target_dataset}_test.csv", out_dir / f"{args.target_dataset}_test.csv")

    print(str(out_dir))


if __name__ == "__main__":
    main()
