#!/usr/bin/env python3
"""Summarize non-claim-valid public on-target smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_TARGETS = {
    "WT": 0.861,
    "ESP": 0.851,
    "HF": 0.865,
    "Sniper-Cas9": 0.935,
    "HL60": 0.402,
}
TRANSFER_TARGETS = {
    "WT_to_HL60": 0.468,
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize non-claim-valid public on-target smoke runs.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument(
        "--smoke-dir",
        default="results/public_benchmarks/smoke",
    )
    ap.add_argument(
        "--output-json",
        default="results/public_benchmarks/smoke/SMOKE_SUMMARY_2026-03-02.json",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    smoke_dir = repo / args.smoke_dir
    summary = {
        "generated": "2026-03-02",
        "status": "non_claim_valid_smoke_only",
        "note": "These are 1-fold, 1-epoch smoke runs to validate the public benchmark path. They are not the final matched public benchmark results.",
        "datasets": {},
        "transfer": {},
    }

    rhos = []
    for path in sorted(smoke_dir.glob("*_fold0_smoke.json")):
        data = load_json(path)
        stem = path.stem.replace("_fold0_smoke", "")
        rho = data.get("gold_rho")
        if stem == "WT_to_HL60":
            target = TRANSFER_TARGETS[stem]
            summary["transfer"][stem] = {
                "rho": rho,
                "target": target,
                "gap": None if rho is None else target - rho,
                "metrics_path": str(path),
            }
            continue
        target = DATASET_TARGETS.get(stem)
        summary["datasets"][stem] = {
            "rho": rho,
            "target": target,
            "gap": None if (rho is None or target is None) else target - rho,
            "metrics_path": str(path),
        }
        if rho is not None:
            rhos.append(rho)

    summary["mean_rho_across_smoke_datasets"] = (sum(rhos) / len(rhos)) if rhos else None
    out_path = repo / args.output_json
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
