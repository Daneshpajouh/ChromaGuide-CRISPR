#!/usr/bin/env python3
"""Validate staged public benchmark inputs and emit a concrete run plan.

This is a lightweight harness: it validates local public benchmark inputs, checks schema,
and emits the exact run order needed for claim-valid benchmarking.
It does not launch expensive training by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ON_TARGET_CANONICAL = [
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
EXPECTED_HEADER = ["sgRNA", "indel"]
CCLMOFF_FILE = "09212024_CCLMoff_dataset.csv"
CCLMOFF_EXPECTED_BYTES = 714692329


def inspect_csv(path: Path) -> dict:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = sum(1 for _ in reader)
    return {
        "path": str(path),
        "header": header,
        "rows": rows,
        "schema_ok": header == EXPECTED_HEADER,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate staged public benchmark inputs and emit a concrete run plan.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument(
        "--output-json",
        default="PUBLIC_BENCHMARK_HARNESS_2026-03-02.json",
        help="Write harness output here.",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    canonical_dir = repo / "data" / "public_benchmarks" / "on_target" / "canonical_9"

    on_target_checks = {}
    all_on_target_ready = True
    for name in ON_TARGET_CANONICAL:
        path = canonical_dir / f"{name}.csv"
        if not path.exists():
            on_target_checks[name] = {"path": str(path), "exists": False, "schema_ok": False, "rows": 0}
            all_on_target_ready = False
            continue
        info = inspect_csv(path)
        info["exists"] = True
        on_target_checks[name] = info
        if not info["schema_ok"] or info["rows"] <= 0:
            all_on_target_ready = False

    off_target_primary_dir = repo / "data" / "public_benchmarks" / "off_target" / "primary_cclmoff"
    cclmoff_file = off_target_primary_dir / CCLMOFF_FILE
    cclmoff_partial = off_target_primary_dir / f"{CCLMOFF_FILE}.partial"
    off_target_secondary_dir = repo / "data" / "public_benchmarks" / "off_target" / "secondary_faststart"
    cclmoff_ready = cclmoff_file.exists() and cclmoff_file.stat().st_size == CCLMOFF_EXPECTED_BYTES

    run_plan = {
        "on_target_single_model_order": [
            {"dataset": name, "input_csv": on_target_checks[name]["path"]}
            for name in ON_TARGET_CANONICAL
        ],
        "after_on_target": [
            "Compute per-dataset SCC/PCC.",
            "Compute 9-dataset mean SCC/PCC.",
            "Run WT-to-HL60 transfer evaluation separately."
        ],
        "off_target_primary_order": [
            "Stage CCLMoff compiled bundle under data/public_benchmarks/off_target/primary_cclmoff.",
            "Run CIRCLE-seq 5-fold CV.",
            "Run CIRCLE-to-GUIDE transfer.",
            "Run DIG-seq and DISCOVER-seq leave-one-dataset-out evaluations."
        ]
    }

    report = {
        "generated": "2026-03-02",
        "on_target": {
            "all_inputs_ready": all_on_target_ready,
            "checks": on_target_checks,
        },
        "off_target": {
            "primary_cclmoff_bundle_present": cclmoff_file.exists(),
            "primary_cclmoff_bundle_complete": cclmoff_ready,
            "primary_cclmoff_actual_bytes": cclmoff_file.stat().st_size if cclmoff_file.exists() else 0,
            "primary_cclmoff_partial_present": cclmoff_partial.exists(),
            "primary_cclmoff_partial_bytes": cclmoff_partial.stat().st_size if cclmoff_partial.exists() else 0,
            "secondary_faststart_present": off_target_secondary_dir.exists(),
            "primary_ready": cclmoff_ready,
        },
        "run_plan": run_plan,
        "claim_note": "This harness validates inputs and the required run order; it does not substitute for matched benchmark training/evaluation results."
    }

    out_path = repo / args.output_json
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
