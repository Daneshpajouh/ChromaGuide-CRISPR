#!/usr/bin/env python3
"""Evaluate local readiness for the frozen public benchmark suites."""

from __future__ import annotations

import argparse
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

OFF_TARGET_PRIMARY = [
    "CIRCLE-seq",
    "GUIDE-seq (Tsai 2015)",
    "CHANGE-seq",
    "DIG-seq",
    "DISCOVER-seq",
    "DISCOVER-seq+",
    "TTISS",
]
CCLMOFF_FILE = "09212024_CCLMoff_dataset.csv"
CCLMOFF_EXPECTED_BYTES = 714692329


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate local readiness for the frozen public benchmark suites.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument(
        "--output-json",
        default="PUBLIC_BENCHMARK_READINESS_2026-03-02.json",
        help="Write report here.",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    local_map_path = repo / "data" / "public_benchmarks" / "acquisition" / "local_dataset_map.json"
    local_map = json.loads(local_map_path.read_text(encoding="utf-8")) if local_map_path.exists() else {}

    on_target_map = ((local_map.get("on_target") or {}).get("canonical_9") or {})
    on_target_available = [name for name in ON_TARGET_CANONICAL if on_target_map.get(name)]
    on_target_missing = [name for name in ON_TARGET_CANONICAL if not on_target_map.get(name)]

    source_map_path = repo / "data" / "public_benchmarks" / "acquisition" / "source_map.json"
    source_map = json.loads(source_map_path.read_text(encoding="utf-8")) if source_map_path.exists() else {}
    repos = (source_map.get("repos") or {})
    off_target_support_repos_ready = all(repos.get(name) for name in ["guideseq", "circleseq", "changeseq", "dagrate_public_data_crisprCas9"])
    cclmoff_file = repo / "data" / "public_benchmarks" / "off_target" / "primary_cclmoff" / CCLMOFF_FILE
    cclmoff_partial = repo / "data" / "public_benchmarks" / "off_target" / "primary_cclmoff" / f"{CCLMOFF_FILE}.partial"
    cclmoff_ready = cclmoff_file.exists() and cclmoff_file.stat().st_size == CCLMOFF_EXPECTED_BYTES

    report = {
        "generated": "2026-03-02",
        "on_target": {
            "full_primary_suite_ready": len(on_target_missing) == 0,
            "available": on_target_available,
            "missing": on_target_missing,
            "separate_dataset_mode": "ready" if len(on_target_missing) == 0 else "blocked",
            "pooled_summary_mode": "ready" if len(on_target_missing) == 0 else "blocked"
        },
        "off_target": {
            "modern_primary_suite_ready": cclmoff_ready,
            "reason": (
                "CCLMoff compiled bundle is fully staged locally; supplement-derived DNABERT-Epi and CHANGE-seq tables are still required for the secondary classifier and uncertainty frames."
                if cclmoff_ready
                else "Primary modern suite still requires the fully downloaded CCLMoff compiled bundle and supplement-derived tables even though supporting repos may be present."
            ),
            "support_repos_ready": off_target_support_repos_ready,
            "cclmoff_dataset_csv": str(cclmoff_file),
            "cclmoff_partial_csv": str(cclmoff_partial),
            "cclmoff_expected_bytes": CCLMOFF_EXPECTED_BYTES,
            "cclmoff_actual_bytes": cclmoff_file.stat().st_size if cclmoff_file.exists() else 0,
            "cclmoff_partial_bytes": cclmoff_partial.stat().st_size if cclmoff_partial.exists() else 0,
            "primary_expected_datasets": OFF_TARGET_PRIMARY,
            "secondary_faststart_ready": bool(repos.get("dagrate_public_data_crisprCas9"))
        },
        "next_steps": [
            "Run single-model public on-target benchmarking now that the canonical nine CSVs are staged locally if on-target is fully ready.",
            "Ensure the full CCLMoff dataset CSV is present at data/public_benchmarks/off_target/primary_cclmoff/09212024_CCLMoff_dataset.csv before any claim-valid off-target run.",
            "Extract DNABERT-Epi supplement tables and CHANGE-seq processed site tables to complete the secondary classifier and uncertainty frames."
        ]
    }

    out_path = repo / args.output_json
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
