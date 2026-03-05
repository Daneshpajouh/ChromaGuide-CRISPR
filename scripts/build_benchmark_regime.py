#!/usr/bin/env python3
"""
Build a stricter benchmark regime with real gene annotations when available.

Sources:
  1) DeepHF-derived tables (existing local CSVs)
  2) CNN-SVR cell-line tables with genomic coordinates

Outputs:
  - Merged benchmark table with harmonized schema
  - Strict splits:
      Split A: gene-held-out (real genes only; no fallback labels)
      Split B: dataset-held-out
      Split C: cell-line-held-out
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import requests


FALLBACK_GENES = {"fallback_gene", "unknown_gene", "placeholder_gene"}
GTF_URL_DEFAULT = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/"
    "gencode.v46.basic.annotation.gtf.gz"
)


@dataclass
class GeneIndex:
    starts: np.ndarray
    ends: np.ndarray
    names: np.ndarray


def normalize_sequence(seq: str, target_len: int = 21) -> str | None:
    s = "".join(ch for ch in str(seq).upper() if ch in {"A", "C", "G", "T", "N"})
    if len(s) < target_len:
        return None
    return s[:target_len]


def canonical_cell_line(name: str) -> str:
    lut = {
        "HELA": "HeLa",
        "HELA_S3": "HeLa",
        "HEK293": "HEK293T",
        "HEK293T": "HEK293T",
        "HCT116": "HCT116",
        "HL60": "HL60",
    }
    k = str(name).strip().upper()
    return lut.get(k, str(name).strip())


def ensure_gtf(gtf_path: Path, url: str) -> None:
    if gtf_path.exists():
        return
    gtf_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading gene annotation: {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(gtf_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"Saved annotation to {gtf_path}")


def parse_gtf_attributes(attr_field: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    parts = [p.strip() for p in attr_field.strip().split(";") if p.strip()]
    for p in parts:
        if " " not in p:
            continue
        k, v = p.split(" ", 1)
        out[k.strip()] = v.strip().strip('"')
    return out


def load_gene_index(gtf_path: Path) -> Dict[str, GeneIndex]:
    print(f"Loading genes from {gtf_path}")
    rows = []
    with gzip.open(gtf_path, "rt") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            chrom, _, feature, start, end, _, _, _, attrs = fields
            if feature != "gene":
                continue
            at = parse_gtf_attributes(attrs)
            gene_name = at.get("gene_name")
            if not gene_name:
                continue
            rows.append((chrom, int(start), int(end), gene_name))
    if not rows:
        raise RuntimeError("No genes parsed from GTF")

    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "gene"])
    out: Dict[str, GeneIndex] = {}
    for chrom, g in df.groupby("chrom"):
        g = g.sort_values("start").reset_index(drop=True)
        out[chrom] = GeneIndex(
            starts=g["start"].to_numpy(np.int64),
            ends=g["end"].to_numpy(np.int64),
            names=g["gene"].to_numpy(object),
        )
    print(f"Loaded {len(df)} genes across {len(out)} chromosomes")
    return out


def map_gene_for_interval(
    chrom: str,
    start: int,
    end: int,
    gene_idx: Dict[str, GeneIndex],
    nearest_max_distance: int = 0,
) -> str:
    idx = gene_idx.get(chrom)
    if idx is None:
        return "unknown_gene"
    mask = (idx.starts <= end) & (idx.ends >= start)
    if not np.any(mask):
        if nearest_max_distance > 0:
            # Distance to interval [start, end] for each gene interval.
            # 0 => overlap; positive => nearest edge distance.
            dist = np.maximum(np.maximum(idx.starts - end, start - idx.ends), 0)
            best_i = int(np.argmin(dist))
            if int(dist[best_i]) <= nearest_max_distance:
                return str(idx.names[best_i])
        return "unknown_gene"
    cand_starts = idx.starts[mask]
    cand_ends = idx.ends[mask]
    overlaps = np.minimum(cand_ends, end) - np.maximum(cand_starts, start) + 1
    best = int(np.argmax(overlaps))
    return str(idx.names[mask][best])


def load_deephf(deephf_dir: Path) -> pd.DataFrame:
    files = {
        "HEK293T": deephf_dir / "DeepHF_HEK293T.csv",
        "HCT116": deephf_dir / "DeepHF_HCT116.csv",
        "HeLa": deephf_dir / "DeepHF_HeLa.csv",
    }
    parts = []
    for cell_line, path in files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path).copy()
        if "sgRNA" in df.columns and "sequence" not in df.columns:
            df = df.rename(columns={"sgRNA": "sequence"})
        if "Efficiency_WT" in df.columns and "efficiency" not in df.columns:
            df = df.rename(columns={"Efficiency_WT": "efficiency"})
        if "Gene" in df.columns and "gene" not in df.columns:
            df = df.rename(columns={"Gene": "gene"})

        if "sequence" not in df.columns or "efficiency" not in df.columns:
            raise ValueError(f"DeepHF file missing columns: {path}")
        if "gene" not in df.columns:
            df["gene"] = "fallback_gene"

        df["sequence"] = df["sequence"].apply(normalize_sequence)
        df = df.dropna(subset=["sequence", "efficiency"]).reset_index(drop=True)

        for i in range(11):
            c = f"feat_{i}"
            if c not in df.columns:
                df[c] = 0.0

        keep = ["sequence", "efficiency", "gene"] + [f"feat_{i}" for i in range(11)]
        out = df[keep].copy()
        out["cell_line"] = canonical_cell_line(cell_line)
        out["dataset"] = "DeepHF"
        parts.append(out)
        print(f"Loaded DeepHF {cell_line}: {len(out)} rows")
    return pd.concat(parts, ignore_index=True)


def frac_char(s: str, ch: str) -> float:
    if not isinstance(s, str) or not s:
        return 0.0
    return float(s.upper().count(ch) / len(s))


def build_cnnsvr_features(df: pd.DataFrame) -> pd.DataFrame:
    seqs = df["sequence"].tolist()
    f0 = [frac_char(s, "G") + frac_char(s, "C") for s in seqs]  # GC fraction
    f1 = [frac_char(s, "A") for s in seqs]
    f2 = [frac_char(s, "T") for s in seqs]
    f3 = [frac_char(s, "C") for s in seqs]
    f4 = [frac_char(s, "G") for s in seqs]
    f5 = [1.0 if str(x).strip() == "+" else 0.0 for x in df["direction"]]
    f6 = [frac_char(s, "A") for s in df["ctcf"].astype(str)]
    f7 = [frac_char(s, "A") for s in df["dnase"].astype(str)]
    f8 = [frac_char(s, "A") for s in df["h3k4me3"].astype(str)]
    f9 = [frac_char(s, "C") for s in df["rrbs"].astype(str)]
    f10 = np.log1p((df["end"] - df["start"]).abs() + 1).astype(float).tolist()

    out = pd.DataFrame(
        {
            "feat_0": f0,
            "feat_1": f1,
            "feat_2": f2,
            "feat_3": f3,
            "feat_4": f4,
            "feat_5": f5,
            "feat_6": f6,
            "feat_7": f7,
            "feat_8": f8,
            "feat_9": f9,
            "feat_10": f10,
        }
    )
    return out


def load_cnnsvr(
    cnnsvr_dir: Path,
    gene_idx: Dict[str, GeneIndex],
    nearest_gene_max_distance: int = 0,
) -> pd.DataFrame:
    files = {
        "HEK293T": cnnsvr_dir / "HEK293T.csv",
        "HCT116": cnnsvr_dir / "HCT116.csv",
        "HeLa": cnnsvr_dir / "HELA.csv",
        "HL60": cnnsvr_dir / "HL60.csv",
    }
    parts = []
    for cell_line, path in files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path).copy()
        required = {"chr", "start", "end", "direction", "seq", "Normalized efficacy", "ctcf", "dnase", "h3k4me3", "rrbs"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")

        df["sequence"] = df["seq"].apply(normalize_sequence)
        df["efficiency"] = pd.to_numeric(df["Normalized efficacy"], errors="coerce")
        df["start"] = pd.to_numeric(df["start"], errors="coerce")
        df["end"] = pd.to_numeric(df["end"], errors="coerce")
        df = df.dropna(subset=["sequence", "efficiency", "start", "end"]).reset_index(drop=True)
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)

        genes = [
            map_gene_for_interval(
                chrom=str(ch),
                start=int(st),
                end=int(en),
                gene_idx=gene_idx,
                nearest_max_distance=nearest_gene_max_distance,
            )
            for ch, st, en in zip(df["chr"], df["start"], df["end"])
        ]
        df["gene"] = genes

        feat_df = build_cnnsvr_features(df)
        out = pd.concat([df[["sequence", "efficiency", "gene"]].reset_index(drop=True), feat_df], axis=1)
        out["cell_line"] = canonical_cell_line(cell_line)
        out["dataset"] = "CNNSVR"
        parts.append(out)
        real_gene_frac = float((~out["gene"].isin(FALLBACK_GENES | {"unknown_gene"})).mean())
        print(f"Loaded CNN-SVR {cell_line}: {len(out)} rows | real_gene_frac={real_gene_frac:.3f}")

    return pd.concat(parts, ignore_index=True)


def save_partition_by_cellline(df: pd.DataFrame, split_dir: Path, partition: str) -> None:
    for cell_line, g in df.groupby("cell_line"):
        path = split_dir / f"{cell_line}_{partition}.csv"
        g.to_csv(path, index=False)


def build_splits(
    merged: pd.DataFrame,
    out_dir: Path,
    seed: int,
    heldout_dataset: str,
    heldout_cell_line: str,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split A: strict gene-held-out using only rows with real gene labels.
    real = merged[
        merged["gene"].notna()
        & ~merged["gene"].isin(FALLBACK_GENES)
        & (merged["gene"] != "unknown_gene")
    ].copy()
    genes = np.array(sorted(real["gene"].unique().tolist()))
    if len(genes) < 10:
        raise RuntimeError(f"Strict Split A requires >=10 genes, found {len(genes)}")
    rng.shuffle(genes)
    n_test = max(1, int(len(genes) * test_frac))
    n_val = max(1, int(len(genes) * val_frac))
    test_genes = set(genes[:n_test])
    val_genes = set(genes[n_test : n_test + n_val])
    train_genes = set(genes[n_test + n_val :])
    split_a = {
        "train": real[real["gene"].isin(train_genes)].copy(),
        "validation": real[real["gene"].isin(val_genes)].copy(),
        "test": real[real["gene"].isin(test_genes)].copy(),
    }
    split_a_dir = out_dir / "split_a_gene_held_out_strict"
    split_a_dir.mkdir(parents=True, exist_ok=True)
    for part, part_df in split_a.items():
        save_partition_by_cellline(part_df, split_a_dir, part)
    (split_a_dir / "metadata.json").write_text(
        json.dumps(
            {
                "split_name": "split_a_gene_held_out_strict",
                "seed": seed,
                "n_rows": {k: int(len(v)) for k, v in split_a.items()},
                "n_genes": int(len(genes)),
                "n_test_genes": int(len(test_genes)),
                "n_val_genes": int(len(val_genes)),
                "n_train_genes": int(len(train_genes)),
                "datasets": sorted(real["dataset"].unique().tolist()),
                "cell_lines": sorted(real["cell_line"].unique().tolist()),
            },
            indent=2,
        )
    )
    print(f"Saved Split A strict: {split_a_dir}")

    # Split B: dataset-held-out.
    if heldout_dataset not in set(merged["dataset"].unique()):
        raise ValueError(f"Held-out dataset '{heldout_dataset}' not found in merged data")
    test_b = merged[merged["dataset"] == heldout_dataset].copy()
    train_pool_b = merged[merged["dataset"] != heldout_dataset].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val_b = max(1, int(len(train_pool_b) * val_frac))
    val_b = train_pool_b.iloc[:n_val_b].copy()
    train_b = train_pool_b.iloc[n_val_b:].copy()

    split_b_dir = out_dir / "split_b_dataset_held_out_strict"
    split_b_dir.mkdir(parents=True, exist_ok=True)
    save_partition_by_cellline(train_b, split_b_dir, "train")
    save_partition_by_cellline(val_b, split_b_dir, "validation")
    save_partition_by_cellline(test_b, split_b_dir, "test")
    (split_b_dir / "metadata.json").write_text(
        json.dumps(
            {
                "split_name": "split_b_dataset_held_out_strict",
                "seed": seed,
                "heldout_dataset": heldout_dataset,
                "n_rows": {"train": int(len(train_b)), "validation": int(len(val_b)), "test": int(len(test_b))},
                "datasets": sorted(merged["dataset"].unique().tolist()),
                "cell_lines": sorted(merged["cell_line"].unique().tolist()),
            },
            indent=2,
        )
    )
    print(f"Saved Split B strict: {split_b_dir}")

    # Split C: cell-line-held-out.
    heldout_cell_line = canonical_cell_line(heldout_cell_line)
    if heldout_cell_line not in set(merged["cell_line"].unique()):
        raise ValueError(f"Held-out cell line '{heldout_cell_line}' not found in merged data")
    test_c = merged[merged["cell_line"] == heldout_cell_line].copy()
    train_pool_c = merged[merged["cell_line"] != heldout_cell_line].sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    n_val_c = max(1, int(len(train_pool_c) * val_frac))
    val_c = train_pool_c.iloc[:n_val_c].copy()
    train_c = train_pool_c.iloc[n_val_c:].copy()

    split_c_dir = out_dir / "split_c_cellline_held_out_strict"
    split_c_dir.mkdir(parents=True, exist_ok=True)
    save_partition_by_cellline(train_c, split_c_dir, "train")
    save_partition_by_cellline(val_c, split_c_dir, "validation")
    save_partition_by_cellline(test_c, split_c_dir, "test")
    (split_c_dir / "metadata.json").write_text(
        json.dumps(
            {
                "split_name": "split_c_cellline_held_out_strict",
                "seed": seed,
                "heldout_cell_line": heldout_cell_line,
                "n_rows": {"train": int(len(train_c)), "validation": int(len(val_c)), "test": int(len(test_c))},
                "datasets": sorted(merged["dataset"].unique().tolist()),
                "cell_lines": sorted(merged["cell_line"].unique().tolist()),
            },
            indent=2,
        )
    )
    print(f"Saved Split C strict: {split_c_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build strict benchmark regime with gene-annotated splits")
    p.add_argument("--deephf_dir", type=str, default="data/deepHF/raw/deepHF")
    p.add_argument("--cnnsvr_dir", type=str, default="data/raw/benchmark_cnnsvr")
    p.add_argument("--gtf_path", type=str, default="data/raw/annotations/gencode.v46.basic.annotation.gtf.gz")
    p.add_argument("--gtf_url", type=str, default=GTF_URL_DEFAULT)
    p.add_argument("--output_merged", type=str, default="data/real/benchmark_merged.csv")
    p.add_argument("--output_splits", type=str, default="data/processed_benchmark")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--heldout_dataset", type=str, default="CNNSVR")
    p.add_argument("--heldout_cell_line", type=str, default="HeLa")
    p.add_argument(
        "--nearest_gene_max_distance",
        type=int,
        default=0,
        help="If >0, map non-overlapping intervals to nearest gene within this distance (bp)",
    )
    args = p.parse_args()

    gtf_path = Path(args.gtf_path)
    ensure_gtf(gtf_path, args.gtf_url)
    gene_idx = load_gene_index(gtf_path)

    deephf_df = load_deephf(Path(args.deephf_dir))
    cnnsvr_df = load_cnnsvr(
        Path(args.cnnsvr_dir),
        gene_idx=gene_idx,
        nearest_gene_max_distance=args.nearest_gene_max_distance,
    )

    merged = pd.concat([deephf_df, cnnsvr_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["sequence", "cell_line", "dataset"]).reset_index(drop=True)
    merged["efficiency"] = pd.to_numeric(merged["efficiency"], errors="coerce")
    merged = merged.dropna(subset=["sequence", "efficiency"]).reset_index(drop=True)

    output_merged = Path(args.output_merged)
    output_merged.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_merged, index=False)
    print(f"Saved merged benchmark: {output_merged}")
    print(
        "Merged stats: "
        f"rows={len(merged)}, genes={merged['gene'].nunique()}, "
        f"datasets={sorted(merged['dataset'].unique().tolist())}, "
        f"cell_lines={sorted(merged['cell_line'].unique().tolist())}"
    )

    build_splits(
        merged=merged,
        out_dir=Path(args.output_splits),
        seed=args.seed,
        heldout_dataset=args.heldout_dataset,
        heldout_cell_line=args.heldout_cell_line,
    )


if __name__ == "__main__":
    main()
