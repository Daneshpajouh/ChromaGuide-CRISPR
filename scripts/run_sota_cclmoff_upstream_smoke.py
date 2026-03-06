#!/usr/bin/env python3
"""Smoke-run upstream CCLMoff model import + single forward pass.

Upstream code imports `from rnafm.fm import pretrained as rnapretrained` while
the pip package exposes `fm`. This script bridges that namespace mismatch only
for runtime compatibility checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument(
        "--data-csv",
        default="data/public_benchmarks/off_target/primary_cclmoff/09212024_CCLMoff_dataset.csv",
    )
    p.add_argument("--output-json", default="results/public_benchmarks/sota_cclmoff_upstream_smoke.json")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    return p.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if raw == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_rnafm_compat() -> None:
    import fm

    # Expose expected alias for older upstream imports.
    if not hasattr(fm.pretrained, "esm1b_rna_t12"):
        fm.pretrained.esm1b_rna_t12 = fm.pretrained.rna_fm_t12

    if "rnafm" not in sys.modules:
        pkg = types.ModuleType("rnafm")
        pkg.fm = fm
        sys.modules["rnafm"] = pkg
    sys.modules["rnafm.fm"] = fm


def first_valid_pair(path: Path) -> tuple[str, str, float]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sg = (row.get("sgRNA_seq") or row.get("sgRNA_type") or "").strip()
            off = (row.get("off_seq") or "").strip()
            if sg and off:
                return sg, off, float(row.get("label", "0"))
    raise RuntimeError("No valid sgRNA/off_seq rows found in data CSV.")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    cclmoff_dir = repo_root / "data" / "public_benchmarks" / "sources" / "CCLMoff"
    data_csv = (repo_root / args.data_csv).resolve()
    out_json = (repo_root / args.output_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    ensure_rnafm_compat()
    if str(cclmoff_dir) not in sys.path:
        sys.path.insert(0, str(cclmoff_dir))

    from my_model import ProtRNA  # type: ignore

    device = resolve_device(args.device)
    model = ProtRNA().to(device)
    model.eval()
    alphabet = model.get_alphabet()
    batch_converter = alphabet.get_batch_converter()

    sg, off, label = first_valid_pair(data_csv)
    seq = (sg + "<sep>" + off).replace("_", "-")
    _, _, toks = batch_converter([("sample0", seq.replace("T", "U"))])
    toks = toks.to(device)

    with torch.no_grad():
        pred = model(toks).detach().cpu().reshape(-1).tolist()[0]

    payload = {
        "model": "CCLMoff_upstream_import_smoke",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "source_dir": str(cclmoff_dir),
        "data_csv": str(data_csv),
        "sample": {
            "sgRNA_seq": sg,
            "off_seq": off,
            "label": label,
            "pred": float(pred),
        },
        "status": "passed",
        "note": "Runtime compatibility smoke only; not a claim-valid benchmark metric.",
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
