#!/usr/bin/env python3
"""Stage acquired public benchmark sources into stable local benchmark paths."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


ON_TARGET_PREFIXES = {
    "WT": "WT",
    "ESP": "ESP",
    "HF": "HF",
    "xCas9": "xCas",
    "SpCas9-NG": "SpCas9-NG",
    "Sniper-Cas9": "Sniper-Cas9",
    "HCT116": "HCT116",
    "HELA": "HELA",
    "HL60": "HL60",
}


def find_first_with_prefix(folder: Path, prefix: str) -> Path | None:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name.startswith(prefix) and path.suffix.lower() == ".csv":
            return path
    return None


def copy_with_conditional_normalization(src: Path, dest: Path) -> dict:
    """Copy CSV and min-max normalize the indel column only if it is not already in [0, 1]."""
    with src.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if not fieldnames or "indel" not in fieldnames:
            shutil.copy2(src, dest)
            return {"normalized": False, "reason": "missing_indel_column"}
        rows = list(reader)

    vals = [float(row["indel"]) for row in rows]
    min_v = min(vals)
    max_v = max(vals)
    needs_norm = min_v < -1e-9 or max_v > 1.0 + 1e-9
    if not needs_norm:
        shutil.copy2(src, dest)
        return {"normalized": False, "min": min_v, "max": max_v}

    span = max_v - min_v
    if span <= 0:
        shutil.copy2(src, dest)
        return {"normalized": False, "reason": "degenerate_span", "min": min_v, "max": max_v}

    for row, value in zip(rows, vals):
        row["indel"] = str((value - min_v) / span)
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "normalized": True,
        "original_min": min_v,
        "original_max": max_v,
        "new_min": 0.0,
        "new_max": 1.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage acquired public benchmark sources into stable local paths.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument(
        "--output-json",
        default="data/public_benchmarks/acquisition/local_dataset_map.json",
        help="Write staged local dataset map here.",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    sources_root = repo / "data" / "public_benchmarks" / "sources"
    staged_root = repo / "data" / "public_benchmarks"

    hnn_datasets = sources_root / "CRISPR_HNN" / "datasets"
    canonical_dest = staged_root / "on_target" / "canonical_9"
    canonical_dest.mkdir(parents=True, exist_ok=True)

    staged = {
        "on_target": {"canonical_9": {}, "raw_mirrors": {}},
        "off_target": {"source_repos": {}, "secondary_faststart": {}},
    }

    if hnn_datasets.exists():
        for canonical_name, prefix in ON_TARGET_PREFIXES.items():
            src = find_first_with_prefix(hnn_datasets, prefix)
            if src is None:
                staged["on_target"]["canonical_9"][canonical_name] = None
                continue
            dest = canonical_dest / f"{canonical_name}.csv"
            norm_info = copy_with_conditional_normalization(src, dest)
            staged["on_target"]["canonical_9"][canonical_name] = {
                "path": str(dest),
                "normalization": norm_info,
            }

    deephf_data = sources_root / "DeepHF" / "data"
    raw_mirror_dest = staged_root / "on_target" / "raw_mirrors"
    raw_mirror_dest.mkdir(parents=True, exist_ok=True)
    if deephf_data.exists():
        for name in ["wt_seq_data_array.pkl", "esp_seq_data_array.pkl", "hf_seq_data_array.pkl"]:
            src = deephf_data / name
            if src.exists():
                dest = raw_mirror_dest / name
                shutil.copy2(src, dest)
                staged["on_target"]["raw_mirrors"][name] = str(dest)

    for repo_name in [
        "dagrate_public_data_crisprCas9",
        "crisporPaper",
        "guideseq",
        "circleseq",
        "changeseq",
    ]:
        src = sources_root / repo_name
        staged["off_target"]["source_repos"][repo_name] = str(src) if src.exists() else None

    dagrate_root = sources_root / "dagrate_public_data_crisprCas9" / "data"
    faststart_dest = staged_root / "off_target" / "secondary_faststart"
    faststart_dest.mkdir(parents=True, exist_ok=True)
    staged_candidates = {
        "Kleinstiver_5gRNA_wholeDataset.csv": dagrate_root / "kleinstiver2015" / "Kleinstiver_5gRNA_wholeDataset.csv",
        "site_seq": dagrate_root / "site_seq",
        "listgarten_elevation_hmg.pkl": dagrate_root / "listgarten_elevation_hmg" / "listgarten_elevation_hmg.pkl",
    }
    for name, src in staged_candidates.items():
        if not src.exists():
            staged["off_target"]["secondary_faststart"][name] = None
            continue
        dest = faststart_dest / name
        if src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)
        staged["off_target"]["secondary_faststart"][name] = str(dest)

    out_path = repo / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(staged, indent=2), encoding="utf-8")
    print(json.dumps(staged, indent=2))


if __name__ == "__main__":
    main()
