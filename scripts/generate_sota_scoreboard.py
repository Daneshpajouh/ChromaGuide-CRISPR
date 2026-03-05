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


def best_on_target_values(summary_files: list[Path]) -> tuple[dict[str, float], dict[str, str]]:
    keys = {
        "mean9_scc": ("mean_9_dataset_progress", None),
        "WT_scc": ("completed_datasets", "WT"),
        "ESP_scc": ("completed_datasets", "ESP"),
        "HF_scc": ("completed_datasets", "HF"),
        "Sniper_Cas9_scc": ("completed_datasets", "Sniper-Cas9"),
        "HL60_scc": ("completed_datasets", "HL60"),
    }

    best_vals: dict[str, float] = {}
    best_src: dict[str, str] = {}

    for path in summary_files:
        data = load_json(path)
        s = data["summary"]

        current = {
            "mean9_scc": float(s["mean_9_dataset_progress"]),
            "WT_scc": float(s["completed_datasets"]["WT"]["mean_gold_rho"]),
            "ESP_scc": float(s["completed_datasets"]["ESP"]["mean_gold_rho"]),
            "HF_scc": float(s["completed_datasets"]["HF"]["mean_gold_rho"]),
            "Sniper_Cas9_scc": float(s["completed_datasets"]["Sniper-Cas9"]["mean_gold_rho"]),
            "HL60_scc": float(s["completed_datasets"]["HL60"]["mean_gold_rho"]),
        }

        for k in keys:
            if k not in best_vals or current[k] > best_vals[k]:
                best_vals[k] = current[k]
                best_src[k] = str(path)

    return best_vals, best_src


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

    on_target_files = [
        repo / "results/public_benchmarks/cluster_harvest_20260305/nibi/full_run_best_FINAL_SUMMARY.json",
        repo / "results/public_benchmarks/cluster_harvest_20260305/rorqual/parallel_full_wave2_FINAL_SUMMARY.json",
        repo / "results/public_benchmarks/cluster_harvest_20260305/fir/full_run_best_FINAL_SUMMARY.json",
    ]
    sweep_files = [
        repo / "results/public_benchmarks/cluster_harvest_20260305/nibi/off_target_manifest_sweep_summary.json",
        repo / "results/public_benchmarks/cluster_harvest_20260305/rorqual/off_target_manifest_sweep_summary.json",
        repo / "results/public_benchmarks/cluster_harvest_20260305/fir/off_target_manifest_sweep_summary.json",
    ]
    uncertainty_files = [
        repo / "results/public_benchmarks/cluster_harvest_20260305/nibi/public_off_target_uncertainty_change_seq.json",
        repo / "results/public_benchmarks/cluster_harvest_20260305/rorqual/public_off_target_uncertainty_change_seq.json",
    ]

    best_on, best_on_src = best_on_target_values(on_target_files)
    transfer_best = float(status["public_on_target"]["best_transfer_wt_to_hl60"]["mean_scc"])
    transfer_src = "PUBLIC_EXECUTION_STATUS_2026-03-05.json"

    rows: list[dict[str, Any]] = []

    # On-target claim-valid metrics
    rows.append(
        metric_record(
            "on_target.mean9_scc",
            "on_target",
            thr["on_target"]["average_thresholds"]["mean_SCC_9_dataset"],
            best_on["mean9_scc"],
            best_on_src["mean9_scc"],
            True,
        )
    )
    rows.append(
        metric_record(
            "on_target.WT_scc",
            "on_target",
            thr["on_target"]["per_dataset_thresholds"]["WT_SCC"],
            best_on["WT_scc"],
            best_on_src["WT_scc"],
            True,
        )
    )
    rows.append(
        metric_record(
            "on_target.ESP_scc",
            "on_target",
            thr["on_target"]["per_dataset_thresholds"]["ESP_SCC"],
            best_on["ESP_scc"],
            best_on_src["ESP_scc"],
            True,
        )
    )
    rows.append(
        metric_record(
            "on_target.HF_scc",
            "on_target",
            thr["on_target"]["per_dataset_thresholds"]["HF_SCC"],
            best_on["HF_scc"],
            best_on_src["HF_scc"],
            True,
        )
    )
    rows.append(
        metric_record(
            "on_target.Sniper_Cas9_scc",
            "on_target",
            thr["on_target"]["per_dataset_thresholds"]["Sniper_Cas9_SCC"],
            best_on["Sniper_Cas9_scc"],
            best_on_src["Sniper_Cas9_scc"],
            True,
        )
    )
    rows.append(
        metric_record(
            "on_target.HL60_scc",
            "on_target",
            thr["on_target"]["per_dataset_thresholds"]["HL60_SCC"],
            best_on["HL60_scc"],
            best_on_src["HL60_scc"],
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
