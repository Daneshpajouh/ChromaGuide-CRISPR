#!/usr/bin/env python3
"""Build trainer-compatible k-fold split directories from staged public on-target CSVs."""

from __future__ import annotations

import argparse
import csv
import random
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


def read_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"sequence": row["sgRNA"], "efficiency": row["indel"]})
    return rows


def enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        enriched = {"sequence": row["sequence"], "efficiency": row["efficiency"]}
        for i in range(11):
            enriched[f"feat_{i}"] = "0.0"
        out.append(enriched)
    return out


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sequence", "efficiency"] + [f"feat_{i}" for i in range(11)]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_folds(rows: list[dict[str, str]], k: int, seed: int) -> list[list[dict[str, str]]]:
    idx = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    fold_sizes = [len(rows) // k] * k
    for i in range(len(rows) % k):
        fold_sizes[i] += 1
    folds = []
    start = 0
    for size in fold_sizes:
        fold_idx = idx[start:start + size]
        folds.append([rows[j] for j in fold_idx])
        start += size
    return folds


def main() -> None:
    ap = argparse.ArgumentParser(description="Build trainer-compatible k-fold split directories from staged public on-target CSVs.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--dataset", default="ALL", help="Canonical dataset name or ALL")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output-root",
        default="data/public_benchmarks/on_target/folds",
        help="Root directory for generated fold split dirs.",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    datasets = CANONICAL if args.dataset == "ALL" else [args.dataset]
    input_root = repo / "data" / "public_benchmarks" / "on_target" / "canonical_9"
    output_root = repo / args.output_root

    summary: dict[str, dict[str, int]] = {}
    for dataset in datasets:
        in_path = input_root / f"{dataset}.csv"
        if not in_path.exists():
            raise FileNotFoundError(in_path)
        rows = enrich_rows(read_rows(in_path))
        folds = build_folds(rows, args.k, args.seed)
        for i in range(args.k):
            fold_dir = output_root / dataset / f"fold_{i}"
            test_rows = folds[i]
            val_rows = folds[(i + 1) % args.k]
            train_rows = [r for j, fold in enumerate(folds) if j not in {i, (i + 1) % args.k} for r in fold]
            write_rows(fold_dir / f"{dataset}_train.csv", train_rows)
            write_rows(fold_dir / f"{dataset}_validation.csv", val_rows)
            write_rows(fold_dir / f"{dataset}_test.csv", test_rows)
        summary[dataset] = {"rows": len(rows), "folds": args.k}

    print(summary)


if __name__ == "__main__":
    main()
