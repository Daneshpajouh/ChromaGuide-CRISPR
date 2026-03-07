#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def metric_record(
    metric_id: str,
    section: str,
    target: float | None,
    best_value: float | None,
    source: str | None,
    claim_valid: bool,
    note: str = "",
) -> dict[str, Any]:
    if target is None or best_value is None:
        return {
            "metric_id": metric_id,
            "section": section,
            "target": target,
            "best_value": best_value,
            "delta_best_minus_target": None,
            "shortfall_to_target": None,
            "pass": False,
            "claim_valid": claim_valid,
            "status": "unavailable",
            "source": source,
            "note": note,
        }

    delta = float(best_value - target)
    return {
        "metric_id": metric_id,
        "section": section,
        "target": float(target),
        "best_value": float(best_value),
        "delta_best_minus_target": delta,
        "shortfall_to_target": float(max(0.0, target - best_value)),
        "pass": bool(delta >= 0.0),
        "claim_valid": claim_valid,
        "status": "scored",
        "source": source,
        "note": note,
    }


def _extract_on_target_candidates(data: dict[str, Any]) -> dict[str, float]:
    current: dict[str, float] = {}

    # Native public benchmark summary shape.
    if "summary" in data and isinstance(data["summary"], dict):
        s = data["summary"]
        if bool(s.get("mean_9_dataset_claim_ready", False)):
            current["mean9_scc"] = float(s["mean_9_dataset_progress"])
        completed = s.get("completed_datasets", {})
        for metric_key, dataset_name in [
            ("WT_scc", "WT"),
            ("ESP_scc", "ESP"),
            ("HF_scc", "HF"),
            ("Sniper_Cas9_scc", "Sniper-Cas9"),
            ("HL60_scc", "HL60"),
        ]:
            ds = completed.get(dataset_name)
            if isinstance(ds, dict) and "mean_gold_rho" in ds:
                current[metric_key] = float(ds["mean_gold_rho"])

    # Upstream threshold summary shape used by reconstructed HNN/FMC runs.
    ds_summaries = data.get("dataset_summaries", {})
    if isinstance(ds_summaries, dict):
        for metric_key, dataset_name in [
            ("WT_scc", "WT"),
            ("ESP_scc", "ESP"),
            ("HF_scc", "HF"),
            ("Sniper_Cas9_scc", "Sniper-Cas9"),
            ("HL60_scc", "HL60"),
        ]:
            ds = ds_summaries.get(dataset_name)
            if isinstance(ds, dict) and "mean_scc" in ds:
                current[metric_key] = float(ds["mean_scc"])

    return current


def best_on_target_values(summary_files: list[Path]) -> tuple[dict[str, float], dict[str, str]]:
    best_vals: dict[str, float] = {}
    best_src: dict[str, str] = {}

    for path in summary_files:
        data = load_json(path)
        current = _extract_on_target_candidates(data)
        if not current:
            continue

        for k, v in current.items():
            if k not in best_vals or current[k] > best_vals[k]:
                best_vals[k] = v
                best_src[k] = str(path)

    return best_vals, best_src


def collect_paths(repo: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for p in repo.glob(pattern):
            rp = p.resolve()
            if rp.exists() and rp not in seen:
                out.append(rp)
                seen.add(rp)
    return out


def best_transfer_value(transfer_fold_files: list[Path], fallback: float | None) -> tuple[float | None, str | None]:
    best_val = fallback
    best_src: str | None = "PUBLIC_EXECUTION_STATUS_2026-03-05.json" if fallback is not None else None
    grouped: dict[Path, list[float]] = {}

    for path in transfer_fold_files:
        data = load_json(path)
        if "gold_rho" in data:
            val = float(data["gold_rho"])
        elif "metrics" in data and "gold_rho" in data["metrics"]:
            val = float(data["metrics"]["gold_rho"])
        elif "gold_spearman" in data:
            val = float(data["gold_spearman"])
        else:
            continue
        run_dir = path.parent
        grouped.setdefault(run_dir, []).append(val)

    for run_dir, vals in grouped.items():
        if len(vals) < 5:
            continue
        mean_val = float(sum(vals) / len(vals))
        if best_val is None or mean_val > best_val:
            best_val = mean_val
            best_src = str(run_dir)

    return best_val, best_src


def best_lodo_by_method(sweep_files: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sweep_files:
        data = load_json(path)
        for split in data.get("splits", []):
            m = split["held_out_method"]
            cand = {
                "auroc": float(split["best_auroc"]),
                "auprc": float(split["best_auprc"]),
                "source": str(path),
            }
            if m not in out or cand["auroc"] + cand["auprc"] > out[m]["auroc"] + out[m]["auprc"]:
                out[m] = cand
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--date", default="2026-03-05")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    output = Path(args.output).resolve() if args.output else repo / f"SOTA_SCOREBOARD_{args.date}.json"

    thr = load_json(repo / "public_claim_thresholds.json")
    status = load_json(repo / "PUBLIC_EXECUTION_STATUS_2026-03-05.json")

    on_target_files = collect_paths(
        repo,
        [
            "results/public_benchmarks/cluster_harvest_*/**/*FINAL_SUMMARY*.json",
            "results/public_benchmarks/cluster_harvest_*/**/*SUMMARY*.json",
            "results/public_benchmarks/*/FINAL_SUMMARY*.json",
            "results/public_benchmarks/**/*SUMMARY*.json",
        ],
    )
    sweep_files = collect_paths(
        repo,
        [
            "results/public_benchmarks/cluster_harvest_*/**/*off_target_manifest_sweep_summary*.json",
            "results/public_benchmarks/**/off_target_manifest_sweep_summary*.json",
        ],
    )
    uncertainty_files = collect_paths(
        repo,
        [
            "results/public_benchmarks/cluster_harvest_*/**/public_off_target_uncertainty*.json",
            "results/public_benchmarks/**/public_off_target_uncertainty*.json",
        ],
    )
    transfer_fold_files = collect_paths(
        repo,
        [
            "results/public_benchmarks/cluster_harvest_*/**/transfer/WT_to_HL60_fold*.json",
            "results/public_benchmarks/**/transfer/WT_to_HL60_fold*.json",
        ],
    )

    best_on, best_on_src = best_on_target_values(on_target_files)
    transfer_fallback = float(status["public_on_target"]["best_transfer_wt_to_hl60"]["mean_scc"])
    transfer_best, transfer_src = best_transfer_value(transfer_fold_files, transfer_fallback)

    rows: list[dict[str, Any]] = []

    # On-target claim-valid metrics
    rows.append(
        metric_record(
            "on_target.mean9_scc",
            "on_target",
            thr["on_target"]["average_thresholds"]["mean_SCC_9_dataset"],
            best_on.get("mean9_scc"),
            best_on_src.get("mean9_scc"),
            True,
        )
    )
    rows.append(
        metric_record(
            "on_target.WT_scc",
            "on_target",
            thr["on_target"]["per_dataset_thresholds"]["WT_SCC"],
            best_on.get("WT_scc"),
            best_on_src.get("WT_scc"),
            True,
        )
    )
    rows.append(
        metric_record(
            "on_target.ESP_scc",
            "on_target",
            thr["on_target"]["per_dataset_thresholds"]["ESP_SCC"],
            best_on.get("ESP_scc"),
            best_on_src.get("ESP_scc"),
            True,
        )
    )
    rows.append(
        metric_record(
            "on_target.HF_scc",
            "on_target",
            thr["on_target"]["per_dataset_thresholds"]["HF_SCC"],
            best_on.get("HF_scc"),
            best_on_src.get("HF_scc"),
            True,
        )
    )
    rows.append(
        metric_record(
            "on_target.Sniper_Cas9_scc",
            "on_target",
            thr["on_target"]["per_dataset_thresholds"]["Sniper_Cas9_SCC"],
            best_on.get("Sniper_Cas9_scc"),
            best_on_src.get("Sniper_Cas9_scc"),
            True,
        )
    )
    rows.append(
        metric_record(
            "on_target.HL60_scc",
            "on_target",
            thr["on_target"]["per_dataset_thresholds"]["HL60_SCC"],
            best_on.get("HL60_scc"),
            best_on_src.get("HL60_scc"),
            True,
        )
    )
    rows.append(
        metric_record(
            "on_target.WT_to_HL60_scc",
            "on_target",
            thr["on_target"]["per_dataset_thresholds"]["WT_to_HL60_SCC"],
            transfer_best,
            transfer_src,
            True,
        )
    )

    # Off-target claim metrics not fully matched yet
    claim_note = "No claim-valid matched frame output yet (blocked by unresolved blank-method provenance / frame parity)."
    rows.extend(
        [
            metric_record(
                "off_target.CIRCLE_seq_CV_AUROC",
                "off_target_classification_transfer_lodo",
                thr["off_target"]["classification_transfer_lodo"]["thresholds"]["CIRCLE_seq_CV_AUROC"],
                None,
                None,
                False,
                claim_note,
            ),
            metric_record(
                "off_target.CIRCLE_seq_CV_AUPRC",
                "off_target_classification_transfer_lodo",
                thr["off_target"]["classification_transfer_lodo"]["thresholds"]["CIRCLE_seq_CV_AUPRC"],
                None,
                None,
                False,
                claim_note,
            ),
            metric_record(
                "off_target.CIRCLE_to_GUIDE_seq_AUROC",
                "off_target_classification_transfer_lodo",
                thr["off_target"]["classification_transfer_lodo"]["thresholds"]["CIRCLE_to_GUIDE_seq_AUROC"],
                None,
                None,
                False,
                claim_note,
            ),
            metric_record(
                "off_target.CIRCLE_to_GUIDE_seq_AUPRC",
                "off_target_classification_transfer_lodo",
                thr["off_target"]["classification_transfer_lodo"]["thresholds"]["CIRCLE_to_GUIDE_seq_AUPRC"],
                None,
                None,
                False,
                claim_note,
            ),
            metric_record(
                "off_target.DIG_seq_LODO_AUROC",
                "off_target_classification_transfer_lodo",
                thr["off_target"]["classification_transfer_lodo"]["thresholds"]["DIG_seq_LODO_AUROC"],
                None,
                None,
                False,
                claim_note,
            ),
            metric_record(
                "off_target.DIG_seq_LODO_AUPRC",
                "off_target_classification_transfer_lodo",
                thr["off_target"]["classification_transfer_lodo"]["thresholds"]["DIG_seq_LODO_AUPRC"],
                None,
                None,
                False,
                claim_note,
            ),
            metric_record(
                "off_target.DISCOVER_seq_plus_LODO_AUROC",
                "off_target_classification_transfer_lodo",
                thr["off_target"]["classification_transfer_lodo"]["thresholds"]["DISCOVER_seq_plus_LODO_AUROC"],
                None,
                None,
                False,
                claim_note,
            ),
            metric_record(
                "off_target.DISCOVER_seq_plus_LODO_AUPRC",
                "off_target_classification_transfer_lodo",
                thr["off_target"]["classification_transfer_lodo"]["thresholds"]["DISCOVER_seq_plus_LODO_AUPRC"],
                None,
                None,
                False,
                claim_note,
            ),
        ]
    )

    # Proxy rows (explicitly not claim-valid)
    best_by_method = best_lodo_by_method(sweep_files)
    if "DISCOVER-seq" in best_by_method:
        rows.append(
            metric_record(
                "proxy.DISCOVER_seq_as_DISCOVER_plus.AUROC",
                "off_target_proxy",
                thr["off_target"]["classification_transfer_lodo"]["thresholds"]["DISCOVER_seq_plus_LODO_AUROC"],
                best_by_method["DISCOVER-seq"]["auroc"],
                best_by_method["DISCOVER-seq"]["source"],
                False,
                "Proxy only; DISCOVER-seq used as stand-in for DISCOVER+ threshold.",
            )
        )
        rows.append(
            metric_record(
                "proxy.DISCOVER_seq_as_DISCOVER_plus.AUPRC",
                "off_target_proxy",
                thr["off_target"]["classification_transfer_lodo"]["thresholds"]["DISCOVER_seq_plus_LODO_AUPRC"],
                best_by_method["DISCOVER-seq"]["auprc"],
                best_by_method["DISCOVER-seq"]["source"],
                False,
                "Proxy only; DISCOVER-seq used as stand-in for DISCOVER+ threshold.",
            )
        )
    if "GUIDE-seq" in best_by_method:
        rows.append(
            metric_record(
                "proxy.GUIDE_seq_as_CIRCLE_to_GUIDE.AUROC",
                "off_target_proxy",
                thr["off_target"]["classification_transfer_lodo"]["thresholds"]["CIRCLE_to_GUIDE_seq_AUROC"],
                best_by_method["GUIDE-seq"]["auroc"],
                best_by_method["GUIDE-seq"]["source"],
                False,
                "Proxy only; held-out GUIDE-seq from LODO is not matched CIRCLE->GUIDE transfer frame.",
            )
        )
        rows.append(
            metric_record(
                "proxy.GUIDE_seq_as_CIRCLE_to_GUIDE.AUPRC",
                "off_target_proxy",
                thr["off_target"]["classification_transfer_lodo"]["thresholds"]["CIRCLE_to_GUIDE_seq_AUPRC"],
                best_by_method["GUIDE-seq"]["auprc"],
                best_by_method["GUIDE-seq"]["source"],
                False,
                "Proxy only; held-out GUIDE-seq from LODO is not matched CIRCLE->GUIDE transfer frame.",
            )
        )

    # Uncertainty
    uncertainty_candidates = [load_json(p) for p in uncertainty_files if p.exists()]
    if uncertainty_candidates:
        best_u = max(uncertainty_candidates, key=lambda x: float(x["metrics"]["test"]["spearman"]))
        best_u_src = next(str(p) for p in uncertainty_files if p.exists() and load_json(p)["metrics"]["test"]["spearman"] == best_u["metrics"]["test"]["spearman"])
        rows.append(
            metric_record(
                "uncertainty.CHANGE_seq_test_spearman",
                "off_target_regression_uncertainty",
                thr["off_target"]["regression_uncertainty"]["thresholds"]["CHANGE_seq_test_Spearman"],
                float(best_u["metrics"]["test"]["spearman"]),
                best_u_src,
                False,
                "Run is on CHANGE-seq proxy table; not claim-valid against frozen crispAI target.",
            )
        )
    else:
        rows.append(
            metric_record(
                "uncertainty.CHANGE_seq_test_spearman",
                "off_target_regression_uncertainty",
                thr["off_target"]["regression_uncertainty"]["thresholds"]["CHANGE_seq_test_Spearman"],
                None,
                None,
                False,
                "No uncertainty result file found.",
            )
        )

    # DNABERT-Epi secondary (not run)
    secondary = thr["off_target"]["unified_classifier_secondary"]["thresholds"]
    rows.extend(
        [
            metric_record("secondary.DNABERT_Epi.PR_AUC", "off_target_unified_secondary", secondary["Lazzarotto_GUIDE_seq_PR_AUC"], None, None, False, "Not run in matched frame."),
            metric_record("secondary.DNABERT_Epi.ROC_AUC", "off_target_unified_secondary", secondary["Lazzarotto_GUIDE_seq_ROC_AUC"], None, None, False, "Not run in matched frame."),
            metric_record("secondary.DNABERT_Epi.F1", "off_target_unified_secondary", secondary["Lazzarotto_GUIDE_seq_F1"], None, None, False, "Not run in matched frame."),
            metric_record("secondary.DNABERT_Epi.MCC", "off_target_unified_secondary", secondary["Lazzarotto_GUIDE_seq_MCC"], None, None, False, "Not run in matched frame."),
        ]
    )

    # Integrated design (not run)
    rows.extend(
        [
            metric_record("integrated.NDCG_at_k", "integrated_design", None, None, None, False, "Not run."),
            metric_record("integrated.Top_k_hit_rate", "integrated_design", None, None, None, False, "Not run."),
            metric_record("integrated.Precision_at_k", "integrated_design", None, None, None, False, "Not run."),
        ]
    )

    claim_valid_rows = [r for r in rows if r["claim_valid"] and r["status"] == "scored"]
    claim_passed = sum(1 for r in claim_valid_rows if r["pass"])

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_date": args.date,
        "repo_root": str(repo),
        "inputs": {
            "thresholds": str(repo / "public_claim_thresholds.json"),
            "status": str(repo / "PUBLIC_EXECUTION_STATUS_2026-03-05.json"),
            "on_target_summaries": [str(p) for p in on_target_files],
            "off_target_lodo_sweeps": [str(p) for p in sweep_files],
            "uncertainty_results": [str(p) for p in uncertainty_files],
        },
        "scoreboard": rows,
        "summary": {
            "total_metrics": len(rows),
            "scored_metrics": sum(1 for r in rows if r["status"] == "scored"),
            "unavailable_metrics": sum(1 for r in rows if r["status"] == "unavailable"),
            "claim_valid_scored_metrics": len(claim_valid_rows),
            "claim_valid_passed": claim_passed,
            "claim_valid_failed": len(claim_valid_rows) - claim_passed,
            "claim_ready": claim_passed == len(claim_valid_rows) and len(claim_valid_rows) > 0,
        },
    }

    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
