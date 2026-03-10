#!/usr/bin/env python3
"""Audit the canonical public DeepHF artifacts and protocol-facing assumptions."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

EXPECTED_DATA = [
    "data/wt_seq_data_array.pkl",
    "data/esp_seq_data_array.pkl",
    "data/hf_seq_data_array.pkl",
]

EXPECTED_MODELS = [
    "models/DeepWt.hd5",
    "models/DeepWt_T7.hd5",
    "models/DeepWt_U6.hd5",
    "models/esp_rnn_model.hd5",
    "models/hf_rnn_model.hd5",
]

TOKEN_MAP = {
    0: "PAD",
    1: "START",
    2: "A",
    3: "T",
    4: "C",
    5: "G",
}


def summarize_pkl(path: Path) -> dict[str, object]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    summary: dict[str, object] = {
        "path": str(path),
        "type": type(obj).__name__,
    }
    if isinstance(obj, (list, tuple)):
        summary["tuple_len"] = len(obj)
        parts = []
        for idx, item in enumerate(obj):
            if hasattr(item, "shape"):
                parts.append({"idx": idx, "shape": list(item.shape), "dtype": str(getattr(item, "dtype", ""))})
            else:
                parts.append({"idx": idx, "type": type(item).__name__})
        summary["parts"] = parts
        if len(obj) >= 1 and hasattr(obj[0], "shape"):
            first = np.asarray(obj[0])
            summary["sequence_tensor_shape"] = list(first.shape)
            if first.ndim >= 2:
                sample = first[0].tolist()
                summary["sample_token_ids_first_row"] = sample[:25]
                summary["token_map"] = TOKEN_MAP
    return summary


def summarize_h5(path: Path) -> dict[str, object]:
    out = {"path": str(path), "datasets": [], "attrs": {}}
    with h5py.File(path, "r") as f:
        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                out["datasets"].append({"name": name, "shape": list(obj.shape), "dtype": str(obj.dtype)})
        f.visititems(walk)
        for k, v in f.attrs.items():
            try:
                out["attrs"][str(k)] = str(v)
            except Exception:
                out["attrs"][str(k)] = "<unserializable>"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--deephf-root", default="data/public_benchmarks/sources/DeepHF")
    ap.add_argument(
        "--output-json",
        default="results/public_benchmarks/deephf_canonical_artifact_audit.json",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    deephf_root = (repo_root / args.deephf_root).resolve()
    out_json = (repo_root / args.output_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "deephf_root": str(deephf_root),
        "expected_data": EXPECTED_DATA,
        "expected_models": EXPECTED_MODELS,
        "present": {},
        "missing": [],
        "data_summaries": {},
        "model_summaries": {},
        "protocol_assumptions": {
            "tokenization": "DeepHF tokenization inferred from repo utilities: PAD=0, START=1, A=2, T=3, C=4, G=5",
            "canonical_use": "Treat original wt/esp/hf_seq_data_array.pkl files and shipped .hd5 models as canonical public artifacts for WT/ESP/HF alignment.",
            "risk_note": "Reprocessed CSV mirrors are convenience artifacts only unless feature parity against the original PKL tensors is demonstrated.",
        },
    }

    for rel in EXPECTED_DATA + EXPECTED_MODELS:
        p = deephf_root / rel
        if p.exists():
            payload["present"][rel] = True
        else:
            payload["missing"].append(rel)

    for rel in EXPECTED_DATA:
        p = deephf_root / rel
        if p.exists():
            payload["data_summaries"][rel] = summarize_pkl(p)

    for rel in EXPECTED_MODELS:
        p = deephf_root / rel
        if p.exists():
            payload["model_summaries"][rel] = summarize_h5(p)

    payload["status"] = "ok" if not payload["missing"] else "missing_artifacts"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
