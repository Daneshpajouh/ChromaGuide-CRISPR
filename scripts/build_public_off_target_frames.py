#!/usr/bin/env python3
"""Build CCLMoff method provenance and frozen public off-target frame manifests."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


UNRESOLVED_METHOD = "UNRESOLVED_BLANK"


def default_method_aliases() -> dict[str, str]:
    return {
        "": UNRESOLVED_METHOD,
        "DIS-seq": "DISCOVER-seq",
        "DISplus-seq": "DISCOVER-seq+",
    }


def build_guide_folds(unique_guides: list[str], fold_count: int, seed: int) -> list[list[str]]:
    guides = list(unique_guides)
    rng = random.Random(seed)
    rng.shuffle(guides)
    folds: list[list[str]] = []
    base = len(guides) // fold_count
    rem = len(guides) % fold_count
    start = 0
    for fold_idx in range(fold_count):
        size = base + (1 if fold_idx < rem else 0)
        end = start + size
        folds.append(guides[start:end])
        start = end
    return folds


def normalize_method(raw: str, aliases: dict[str, str]) -> str:
    key = (raw or "").strip()
    if key in aliases:
        return aliases[key]
    return key if key else UNRESOLVED_METHOD


def scan_cclmoff(data_path: Path, aliases: dict[str, str]) -> dict:
    raw_stats: dict[str, dict] = defaultdict(lambda: {"rows": 0, "pos": 0, "neg": 0, "guides": set()})
    canonical_stats: dict[str, dict] = defaultdict(lambda: {"rows": 0, "pos": 0, "neg": 0, "guides": set()})

    with data_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_method = (row.get("Method") or "").strip()
            canonical = normalize_method(raw_method, aliases)
            label = 1 if float(row["label"]) > 0.0 else 0
            guide = row.get("sgRNA_type", "") or row.get("sgRNA_seq", "")

            raw_stats[raw_method]["rows"] += 1
            canonical_stats[canonical]["rows"] += 1
            if label > 0:
                raw_stats[raw_method]["pos"] += 1
                canonical_stats[canonical]["pos"] += 1
            else:
                raw_stats[raw_method]["neg"] += 1
                canonical_stats[canonical]["neg"] += 1
            if guide:
                raw_stats[raw_method]["guides"].add(guide)
                canonical_stats[canonical]["guides"].add(guide)

    return {"raw": raw_stats, "canonical": canonical_stats}


def to_method_entries(raw_stats: dict[str, dict], aliases: dict[str, str]) -> list[dict]:
    entries: list[dict] = []
    for raw_method, stats in raw_stats.items():
        normalized = normalize_method(raw_method, aliases)
        entries.append(
            {
                "raw_method": raw_method,
                "normalized_method": normalized,
                "row_count": stats["rows"],
                "positive_count": stats["pos"],
                "negative_count": stats["neg"],
                "unique_guides": len(stats["guides"]),
                "claim_valid_allowed": normalized != UNRESOLVED_METHOD,
                "note": (
                    "Blank Method rows are unresolved and excluded from claim-valid frames until verified."
                    if normalized == UNRESOLVED_METHOD
                    else ""
                ),
            }
        )
    entries.sort(key=lambda item: item["row_count"], reverse=True)
    return entries


def ready_manifest_common(data_path: Path, method_map_path: Path) -> dict:
    return {
        "data_path": str(data_path),
        "method_map_json": str(method_map_path),
        "seed": 42,
    }


def build_circle_cv_manifest(data_path: Path, method_map_path: Path, canonical_stats: dict[str, dict]) -> dict:
    stats = canonical_stats.get("CIRCLE-seq")
    if not stats or stats["rows"] == 0:
        return {
            "frame_name": "cclmoff_circle_cv",
            "status": "blocked",
            "blocked_reason": (
                "No resolved CIRCLE-seq rows were found in the staged CCLMoff CSV. "
                "The large blank Method bucket remains unresolved and excluded."
            ),
            "split_mode": "guide_kfold",
            "fold_count": 5,
            "include_methods": ["CIRCLE-seq"],
            **ready_manifest_common(data_path, method_map_path),
        }

    guides = sorted(stats["guides"])
    folds = build_guide_folds(guides, 5, 42)
    return {
        "frame_name": "cclmoff_circle_cv",
        "status": "ready",
        "claim_valid_for_frozen_thresholds": True,
        "split_mode": "guide_kfold",
        "fold_count": 5,
        "include_methods": ["CIRCLE-seq"],
        "exclude_methods": [UNRESOLVED_METHOD],
        "guide_folds": folds,
        **ready_manifest_common(data_path, method_map_path),
    }


def build_circle_to_guide_manifest(data_path: Path, method_map_path: Path, canonical_stats: dict[str, dict]) -> dict:
    circle_ready = canonical_stats.get("CIRCLE-seq", {}).get("rows", 0) > 0
    guide_ready = canonical_stats.get("GUIDE-seq", {}).get("rows", 0) > 0
    if not (circle_ready and guide_ready):
        return {
            "frame_name": "cclmoff_circle_to_guide",
            "status": "blocked",
            "blocked_reason": "Requires both resolved CIRCLE-seq and GUIDE-seq rows in the CCLMoff CSV.",
            "split_mode": "train_test_methods",
            "include_methods": ["CIRCLE-seq", "GUIDE-seq"],
            "train_methods": ["CIRCLE-seq"],
            "test_methods": ["GUIDE-seq"],
            **ready_manifest_common(data_path, method_map_path),
        }
    return {
        "frame_name": "cclmoff_circle_to_guide",
        "status": "ready",
        "claim_valid_for_frozen_thresholds": True,
        "split_mode": "train_test_methods",
        "include_methods": ["CIRCLE-seq", "GUIDE-seq"],
        "exclude_methods": [UNRESOLVED_METHOD],
        "train_methods": ["CIRCLE-seq"],
        "test_methods": ["GUIDE-seq"],
        **ready_manifest_common(data_path, method_map_path),
    }


def build_lodo_manifest(data_path: Path, method_map_path: Path, canonical_stats: dict[str, dict]) -> dict:
    stable_methods = []
    for method_name, stats in canonical_stats.items():
        if method_name == UNRESOLVED_METHOD:
            continue
        if stats["pos"] >= 30 and stats["neg"] > 0:
            stable_methods.append(method_name)
    stable_methods = sorted(stable_methods)

    if len(stable_methods) < 2:
        return {
            "frame_name": "cclmoff_lodo",
            "status": "blocked",
            "blocked_reason": "Need at least two resolved methods with >=30 positives and >=1 negative for LODO.",
            "split_mode": "lodo",
            "candidate_methods": stable_methods,
            **ready_manifest_common(data_path, method_map_path),
        }

    splits = []
    for held_out in stable_methods:
        train_methods = [item for item in stable_methods if item != held_out]
        splits.append(
            {
                "held_out_method": held_out,
                "train_methods": train_methods,
                "test_methods": [held_out],
            }
        )

    required_for_frozen = {"GUIDE-seq", "DISCOVER-seq+"}
    return {
        "frame_name": "cclmoff_lodo",
        "status": "ready",
        "claim_valid_for_frozen_thresholds": required_for_frozen.issubset(set(stable_methods)),
        "split_mode": "lodo",
        "include_methods": stable_methods,
        "exclude_methods": [UNRESOLVED_METHOD],
        "candidate_methods": stable_methods,
        "splits": splits,
        "note": "DIG-seq is not present as a resolved method name in the staged CSV; frozen DIG-seq thresholds remain unmatched.",
        **ready_manifest_common(data_path, method_map_path),
    }


def build_crispai_manifest(frames_dir: Path) -> dict:
    change_seq_path = frames_dir.parent / "secondary_change_seq" / "CHANGE_seq_processed_table.csv"
    provenance_path = frames_dir.parent / "secondary_change_seq" / "CHANGE_seq_processed_table.provenance.json"
    status = "ready" if change_seq_path.exists() else "blocked"
    payload = {
        "frame_name": "crispai_change_regression_uncertainty",
        "status": status,
        "data_path": str(change_seq_path),
        "provenance_path": str(provenance_path),
        "split_mode": "guide_holdout",
        "split_recipe": {
            "train_fraction": 0.70,
            "validation_fraction": 0.20,
            "test_fraction": 0.10,
            "seed": 42,
            "group_key": "sgRNA_id",
        },
        "task_type": "regression_uncertainty",
        "source_of_truth": "Processed CHANGE-seq table required by crispAI; do not substitute the CCLMoff CSV unless continuous activity targets are verified.",
    }
    if status != "ready":
        payload["blocked_reason"] = (
            "Processed CHANGE-seq table is not staged at data/public_benchmarks/off_target/secondary_change_seq/"
            "CHANGE_seq_processed_table.csv."
        )
        payload["claim_valid_for_frozen_thresholds"] = False
        return payload

    payload["claim_valid_for_frozen_thresholds"] = True
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        payload["provenance"] = provenance
        if provenance.get("is_proxy_from_cclmoff"):
            payload["status"] = "ready_proxy"
            payload["claim_valid_for_frozen_thresholds"] = False
            payload["note"] = (
                "CHANGE-seq table is staged as a proxy extracted from the CCLMoff bundle. "
                "Useful for uncertainty experiments but not claim-valid against crispAI frozen targets."
            )
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="data/public_benchmarks/off_target/primary_cclmoff/09212024_CCLMoff_dataset.csv",
    )
    parser.add_argument(
        "--method-map-out",
        default="data/public_benchmarks/off_target/primary_cclmoff/method_map.json",
    )
    parser.add_argument(
        "--frames-dir",
        default="data/public_benchmarks/off_target/frames",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    method_map_path = Path(args.method_map_out).resolve()
    frames_dir = Path(args.frames_dir).resolve()

    aliases = default_method_aliases()
    scan = scan_cclmoff(data_path, aliases)
    raw_stats = scan["raw"]
    canonical_stats = scan["canonical"]

    method_map_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(data_path),
        "unresolved_method_name": UNRESOLVED_METHOD,
        "policy": {
            "blank_method_rows": "excluded_from_claim_valid_frames_until_verified",
            "aliasing": aliases,
        },
        "methods": to_method_entries(raw_stats, aliases),
        "canonical_method_summary": [
            {
                "normalized_method": method_name,
                "row_count": stats["rows"],
                "positive_count": stats["pos"],
                "negative_count": stats["neg"],
                "unique_guides": len(stats["guides"]),
            }
            for method_name, stats in sorted(
                canonical_stats.items(),
                key=lambda item: item[1]["rows"],
                reverse=True,
            )
        ],
    }
    write_json(method_map_path, method_map_payload)

    frame_payloads = {
        "cclmoff_circle_cv.json": build_circle_cv_manifest(data_path, method_map_path, canonical_stats),
        "cclmoff_circle_to_guide.json": build_circle_to_guide_manifest(data_path, method_map_path, canonical_stats),
        "cclmoff_lodo.json": build_lodo_manifest(data_path, method_map_path, canonical_stats),
        "crispai_change_regression_uncertainty.json": build_crispai_manifest(frames_dir),
    }
    for filename, payload in frame_payloads.items():
        write_json(frames_dir / filename, payload)

    inventory = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frames_dir": str(frames_dir),
        "files": sorted(frame_payloads),
    }
    write_json(frames_dir / "frame_inventory.json", inventory)

    print(f"Wrote method map: {method_map_path}")
    for filename in sorted(frame_payloads):
        print(f"Wrote frame: {frames_dir / filename}")


if __name__ == "__main__":
    main()
