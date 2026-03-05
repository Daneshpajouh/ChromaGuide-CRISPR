#!/usr/bin/env python3
"""Compare current local status against frozen public SOTA targets.

This is a claim-hygiene helper. It does not pretend internal private-split metrics
are directly comparable to public benchmarks. Instead, it builds a structured gap
report showing:

1. current internal status,
2. the frozen public target thresholds,
3. which public benchmark outputs are still missing,
4. what would need to be measured before an external SOTA claim is valid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compare_internal_vs_public(internal: dict, public_targets: dict) -> dict:
    best_nonstacked = internal.get("best_splitA_nonstacked_ensemble", {}) or {}
    best_single = internal.get("best_splitA_nonstacked_single", {}) or {}
    offt = internal.get("offtarget_metrics", {}) or {}

    report: dict = {
        "internal_status_snapshot": {
            "best_splitA_nonstacked_single_rho": best_single.get("rho"),
            "best_splitA_nonstacked_ensemble_rho": best_nonstacked.get("rho"),
            "strict_splitA_best_single_rho": (internal.get("best_splitA_strict_single", {}) or {}).get("rho"),
            "offtarget_pair_aware_test_auroc": offt.get("pair_aware_test_auroc"),
            "offtarget_pair_aware_test_auprc": offt.get("pair_aware_test_auprc"),
            "conformal_empirical_coverage": internal.get("conformal_empirical_coverage"),
        },
        "public_target_snapshot": {
            "on_target_leader_status": public_targets["on_target"]["leader_status"],
            "on_target_average_target_leader": public_targets["on_target"]["leaders"]["average_target_leader"],
            "on_target_per_dataset_target_leader": public_targets["on_target"]["leaders"]["per_dataset_and_transfer_leader"],
            "on_target_mean_scc_target": public_targets["on_target"]["average_targets"]["mean_SCC_9_dataset"],
            "on_target_mean_pcc_target": public_targets["on_target"]["average_targets"]["mean_PCC_9_dataset"],
            "off_target_leader_single_model": public_targets["off_target"]["leader_single_model"],
            "off_target_circle_to_guide_auroc_target": public_targets["off_target"]["datasets"]["CIRCLE_to_GUIDE-seq"]["metrics"]["AUROC"],
            "off_target_circle_to_guide_auprc_target": public_targets["off_target"]["datasets"]["CIRCLE_to_GUIDE-seq"]["metrics"]["AUPRC"],
        },
        "claim_validity": {
            "external_sota_claim_valid_now": False,
            "reason": "No matched public benchmark outputs are present in the current local status artifact. Internal Split A and matched held-out results are not directly comparable to public nine-dataset or multi-assay benchmark regimes."
        },
        "missing_public_measurements": {
            "on_target_average_target": {
                "leader": public_targets["on_target"]["average_targets"]["leader"],
                "mean_SCC_9_dataset": public_targets["on_target"]["average_targets"]["mean_SCC_9_dataset"],
                "mean_PCC_9_dataset": public_targets["on_target"]["average_targets"]["mean_PCC_9_dataset"],
            },
            "on_target_nine_dataset_cv": [],
            "on_target_transfer": [],
            "off_target_public": [
                "CIRCLE-seq CV",
                "CIRCLE->GUIDE-seq",
                "DIG-seq LODO",
                "DISCOVER-seq/DISCOVER-seq+ LODO"
            ],
            "calibration": [
                "public comparable calibration metrics vs external literature"
            ],
            "integrated_design": [
                "explicit public ranking benchmark with NDCG@k / Top-k utility"
            ]
        },
        "next_required_actions": [
            "Produce public on-target results on the frozen nine-dataset benchmark in both separate-dataset and pooled-summary forms.",
            "Run single-model public on-target evaluation before ensemble/stacked public evaluation.",
            "Produce public off-target results on the CCLMoff-style assay-derived benchmark suite.",
            "Run the narrower remaining-gap literature pass before making any final external SOTA claim.",
            "Keep single-model and ensemble/stacked results separated in all reports."
        ]
    }

    on_target_targets = public_targets["on_target"]["per_dataset_targets"]
    for dataset_name, meta in on_target_targets.items():
        if dataset_name == "leader":
            continue
        report["missing_public_measurements"]["on_target_nine_dataset_cv"].append(
            {
                "dataset": dataset_name,
                "metric": meta["metric"],
                "target_to_beat": meta["target_value"],
            }
        )
    for target_name, meta in public_targets["on_target"]["transfer_targets"].items():
        if target_name == "leader":
            continue
        report["missing_public_measurements"]["on_target_transfer"].append(
            {
                "evaluation": target_name,
                "metric": meta["metric"],
                "target_to_beat": meta["target_value"],
            }
        )

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare current local status against frozen public SOTA targets.")
    ap.add_argument(
        "--internal-status",
        default="results/runs/w49_proposal_status_latest.json",
        help="Current internal status JSON.",
    )
    ap.add_argument(
        "--public-targets",
        default="PUBLIC_SOTA_TARGETS_2026-03-02.json",
        help="Frozen public target manifest JSON.",
    )
    ap.add_argument(
        "--output-json",
        default="PUBLIC_SOTA_GAP_ANALYSIS_2026-03-02.json",
        help="Output JSON report.",
    )
    args = ap.parse_args()

    internal_path = Path(args.internal_status)
    public_path = Path(args.public_targets)
    out_path = Path(args.output_json)

    internal = load_json(internal_path)
    public_targets = load_json(public_path)
    report = compare_internal_vs_public(internal, public_targets)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
