#!/usr/bin/env python3
"""
Prepare merged DeepHF-style CSV data for ChromaGuide training.

This script consolidates per-cell-line CSV files into one table with a stable schema:
  - required: sequence, efficiency, gene
  - optional: feat_0 ... feat_n
  - added: cell_line
"""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


RENAME_MAP = {
    "sgRNA": "sequence",
    "Efficiency_WT": "efficiency",
    "Gene": "gene",
}


def _find_cellline_files(raw_dir: Path) -> Dict[str, Path]:
    return {
        "HEK293T": raw_dir / "DeepHF_HEK293T.csv",
        "HCT116": raw_dir / "DeepHF_HCT116.csv",
        "HeLa": raw_dir / "DeepHF_HeLa.csv",
    }


def _standardize_columns(df: pd.DataFrame, cell_line: str, allow_gene_fallback: bool) -> pd.DataFrame:
    df = df.rename(columns=RENAME_MAP).copy()

    required = {"sequence", "efficiency"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for {cell_line}: {missing}")

    if "gene" not in df.columns:
        if not allow_gene_fallback:
            raise ValueError(
                f"Column 'gene' missing for {cell_line}. "
                "Run with --allow-gene-fallback only for non-gene-held-out experiments."
            )
        df["gene"] = f"{cell_line}_fallback_gene"

    # Keep feat_* columns when available.
    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    keep_cols = ["sequence", "efficiency", "gene"] + feat_cols
    out = df[keep_cols].copy()
    out["cell_line"] = cell_line
    return out


def load_deephf_data(raw_dir: Path, allow_gene_fallback: bool) -> pd.DataFrame:
    files = _find_cellline_files(raw_dir)
    parts: List[pd.DataFrame] = []

    for cell_line, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")
        df = pd.read_csv(path)
        std = _standardize_columns(df, cell_line, allow_gene_fallback=allow_gene_fallback)
        parts.append(std)
        print(f"Loaded {cell_line}: {len(std)} rows from {path}")

    merged = pd.concat(parts, ignore_index=True)
    merged = merged.drop_duplicates(subset=["sequence", "cell_line"]).reset_index(drop=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare merged DeepHF CSV for ChromaGuide")
    parser.add_argument("--raw-dir", type=str, default="data/deepHF/raw/deepHF")
    parser.add_argument("--output", type=str, default="data/real/merged.csv")
    parser.add_argument(
        "--allow-gene-fallback",
        action="store_true",
        help="Allow synthetic gene labels when raw data lacks gene column",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    merged = load_deephf_data(raw_dir, allow_gene_fallback=args.allow_gene_fallback)
    merged.to_csv(output, index=False)

    unique_genes = merged["gene"].nunique()
    print(f"Saved merged dataset: {output}")
    print(f"Rows={len(merged)} | Cell lines={merged['cell_line'].nunique()} | Genes={unique_genes}")

    if unique_genes < 5:
        print(
            "WARNING: fewer than 5 unique genes detected; strict Split A gene-held-out "
            "will fail unless real gene annotations are provided."
        )


if __name__ == "__main__":
    main()
