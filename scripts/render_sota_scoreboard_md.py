#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SECTION_ORDER = [
    "on_target",
    "off_target_classification_transfer_lodo",
    "off_target_proxy",
    "off_target_regression_uncertainty",
    "off_target_unified_secondary",
    "integrated_design",
]

SECTION_TITLES = {
    "on_target": "On-Target",
    "off_target_classification_transfer_lodo": "Off-Target Primary",
    "off_target_proxy": "Off-Target Proxy",
    "off_target_regression_uncertainty": "Uncertainty",
    "off_target_unified_secondary": "Secondary Unified Classifier",
    "integrated_design": "Integrated Design",
}


def fmt_num(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        return f"{value:.10f}".rstrip("0").rstrip(".")
    return str(value)


def pass_text(row: dict[str, Any]) -> str:
    if row["status"] == "unavailable":
        return "NA"
    return "Yes" if row["pass"] else "No"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()

    input_json = Path(args.input_json).resolve()
    output_md = Path(args.output_md).resolve()
    payload = json.loads(input_json.read_text())
    rows = payload["scoreboard"]

    sections: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sections.setdefault(row["section"], []).append(row)

    lines: list[str] = []
    lines.append(f"# SOTA Scoreboard ({payload['as_of_date']})")
    lines.append("")
    lines.append(f"- Generated at: `{payload['generated_at_utc']}`")
    lines.append(f"- Claim-valid passed: `{payload['summary']['claim_valid_passed']}` / `{payload['summary']['claim_valid_scored_metrics']}`")
    lines.append(f"- Claim-ready: `{'yes' if payload['summary']['claim_ready'] else 'no'}`")
    lines.append("")

    for section in SECTION_ORDER:
        section_rows = sections.get(section, [])
        if not section_rows:
            continue
        lines.append(f"## {SECTION_TITLES[section]}")
        lines.append("")
        lines.append("| Metric | Best | Target | Gap (best-target) | Pass | Claim-valid | Status |")
        lines.append("|---|---:|---:|---:|---|---|---|")
        for row in section_rows:
            lines.append(
                "| {metric} | {best} | {target} | {gap} | {passed} | {claim} | {status} |".format(
                    metric=row["metric_id"],
                    best=fmt_num(row["best_value"]),
                    target=fmt_num(row["target"]),
                    gap=fmt_num(row["delta_best_minus_target"]),
                    passed=pass_text(row),
                    claim="Yes" if row["claim_valid"] else "No",
                    status=row["status"],
                )
            )
        lines.append("")

        notes = [f"- `{row['metric_id']}`: {row['note']}" for row in section_rows if row.get("note")]
        if notes:
            lines.append("Notes:")
            lines.extend(notes)
            lines.append("")

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md)


if __name__ == "__main__":
    main()
