#!/usr/bin/env python3
"""Assess readiness for the frozen public on-target benchmark.

This does not run training. It answers:
1. Which canonical public datasets are available locally?
2. Which are missing?
3. What separate-dataset and pooled evaluation modes should be run?
4. Whether a claim-valid public on-target run can execute immediately.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CANONICAL_DATASETS = {
    "WT": "data/public_benchmarks/on_target/canonical_9/WT.csv",
    "ESP": "data/public_benchmarks/on_target/canonical_9/ESP.csv",
    "HF": "data/public_benchmarks/on_target/canonical_9/HF.csv",
    "xCas9": "data/public_benchmarks/on_target/canonical_9/xCas9.csv",
    "SpCas9-NG": "data/public_benchmarks/on_target/canonical_9/SpCas9-NG.csv",
    "Sniper-Cas9": "data/public_benchmarks/on_target/canonical_9/Sniper-Cas9.csv",
    "HCT116": "data/public_benchmarks/on_target/canonical_9/HCT116.csv",
    "HELA": "data/public_benchmarks/on_target/canonical_9/HELA.csv",
    "HL60": "data/public_benchmarks/on_target/canonical_9/HL60.csv",
}


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        # subtract header if present
        rows = list(reader)
    return max(len(rows) - 1, 0) if rows else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Assess readiness for the frozen public on-target benchmark.")
    ap.add_argument(
        "--repo-root",
        default=".",
        help="Repository root (defaults to current working directory).",
    )
    ap.add_argument(
        "--public-targets",
        default="PUBLIC_SOTA_TARGETS_2026-03-02.json",
        help="Frozen public target manifest.",
    )
    ap.add_argument(
        "--output-json",
        default="PUBLIC_ON_TARGET_READINESS_2026-03-02.json",
        help="Output readiness report.",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    public_targets = json.loads((repo / args.public_targets).read_text(encoding="utf-8"))

    available = []
    missing = []
    dataset_status = []

    for name, rel in CANONICAL_DATASETS.items():
        path = repo / rel
        if path.exists():
            rows = count_csv_rows(path)
            available.append(name)
            dataset_status.append(
                {
                    "dataset": name,
                    "available_locally": True,
                    "path": rel,
                    "rows": rows,
                }
            )
        else:
            missing.append(name)
            dataset_status.append(
                {
                    "dataset": name,
                    "available_locally": False,
                    "path": rel,
                    "rows": None,
                }
            )

    next_actions = []
    if missing:
        next_actions.append("Acquire the missing public on-target datasets locally before claiming a full public benchmark run.")
    else:
        next_actions.append("The full canonical public on-target suite is now staged locally and ready for matched evaluation.")
    next_actions.extend(
        [
            "Run the current best single non-stacked model on each public dataset separately under matched protocol.",
            "Report pooled summary only after all separate-dataset outputs are produced.",
            "After the canonical suite is complete, run any additional public datasets as secondary evidence in the same separate-then-pooled style.",
            "Do not treat scripts/download_deepHF_data.py output as canonical nine-dataset coverage; it only yields partial local DeepHF-derived tables, not the full frozen public suite."
        ]
    )

    report = {
        "as_of_date": "2026-03-02",
        "frozen_public_on_target_regime": public_targets["on_target"]["frozen_benchmark_regime"],
        "leader_status": public_targets["on_target"]["leader_status"],
        "current_local_readiness": {
            "claim_valid_full_public_run_possible_now": len(missing) == 0,
            "available_dataset_count": len(available),
            "missing_dataset_count": len(missing),
            "available_datasets": available,
            "missing_datasets": missing,
            "dataset_status": dataset_status,
        },
        "required_evaluation_modes": {
            "separate_dataset_mode": {
                "required": True,
                "description": "Run one matched evaluation per public dataset and report per-dataset SCC/PCC.",
                "status_now": "blocked" if len(missing) > 0 else "ready",
            },
            "pooled_summary_mode": {
                "required": True,
                "description": "Aggregate separate-dataset outputs into frozen public summary metrics (9-dataset mean plus transfer-specific metrics).",
                "status_now": "blocked" if len(missing) > 0 else "ready",
            },
        },
        "minimum_next_actions": next_actions,
    }

    out_path = repo / args.output_json
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
