#!/usr/bin/env python3
"""Stage a CHANGE-seq proxy table for regression/uncertainty benchmarking.

This extractor uses rows where Method == CHANGE-seq from the staged CCLMoff CSV.
It writes a compact processed table plus provenance metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="data/public_benchmarks/off_target/primary_cclmoff/09212024_CCLMoff_dataset.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="data/public_benchmarks/off_target/secondary_change_seq/CHANGE_seq_processed_table.csv",
    )
    parser.add_argument(
        "--output-provenance",
        default="data/public_benchmarks/off_target/secondary_change_seq/CHANGE_seq_processed_table.provenance.json",
    )
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    in_path = Path(args.input_csv).resolve()
    out_path = Path(args.output_csv).resolve()
    prov_path = Path(args.output_provenance).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "sgRNA_id",
        "sgRNA_seq",
        "off_seq",
        "chr",
        "location",
        "strand",
        "length",
        "label_binary",
        "activity_raw_read",
        "activity_log1p_read",
        "method",
        "source_row_id",
    ]

    total_rows = 0
    written_rows = 0
    positive_rows = 0
    nonzero_activity_rows = 0
    max_activity = 0.0

    with in_path.open("r", encoding="utf-8", errors="replace", newline="") as handle_in:
        reader = csv.DictReader(handle_in)
        with out_path.open("w", encoding="utf-8", newline="") as handle_out:
            writer = csv.DictWriter(handle_out, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                total_rows += 1
                if (row.get("Method") or "").strip() != "CHANGE-seq":
                    continue

                raw_read = float(row.get("read", "0") or 0.0)
                label = 1 if float(row.get("label", "0") or 0.0) > 0 else 0
                writer.writerow(
                    {
                        "sgRNA_id": row.get("sgRNA_type", "") or row.get("sgRNA_seq", ""),
                        "sgRNA_seq": row.get("sgRNA_seq", ""),
                        "off_seq": row.get("off_seq", ""),
                        "chr": row.get("chr", ""),
                        "location": row.get("Location", ""),
                        "strand": row.get("Direction", ""),
                        "length": row.get("Length", ""),
                        "label_binary": label,
                        "activity_raw_read": raw_read,
                        "activity_log1p_read": math.log1p(raw_read),
                        "method": "CHANGE-seq",
                        "source_row_id": row.get("id", ""),
                    }
                )
                written_rows += 1
                if label > 0:
                    positive_rows += 1
                if raw_read > 0.0:
                    nonzero_activity_rows += 1
                if raw_read > max_activity:
                    max_activity = raw_read
                if args.max_rows > 0 and written_rows >= args.max_rows:
                    break

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(in_path),
        "output_csv": str(out_path),
        "is_proxy_from_cclmoff": True,
        "method_filter": "CHANGE-seq",
        "notes": [
            "This table is extracted from the staged CCLMoff bundle.",
            "Use for exploratory regression/uncertainty runs.",
            "Not claim-valid against crispAI frozen targets without primary-source parity verification.",
        ],
        "stats": {
            "total_rows_scanned": total_rows,
            "rows_written": written_rows,
            "positive_rows": positive_rows,
            "nonzero_activity_rows": nonzero_activity_rows,
            "max_activity_raw_read": max_activity,
        },
    }
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps(provenance, indent=2))

    print(f"Wrote {written_rows} rows to {out_path}")
    print(f"Wrote provenance: {prov_path}")


if __name__ == "__main__":
    main()
