#!/usr/bin/env python3
"""Run crispAI upstream offt-score smoke and capture blocker details."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--python-bin", default="/Users/studio/.pyenv/versions/3.10.19/bin/python")
    p.add_argument(
        "--input-file",
        default="data/public_benchmarks/sources/crispAI_crispr-offtarget-uncertainty/crispAI_score/example_offt_input.txt",
    )
    p.add_argument("--output-csv", default="results/public_benchmarks/sota_crispai_smoke.csv")
    p.add_argument("--output-json", default="results/public_benchmarks/sota_crispai_upstream_smoke.json")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--n-mismatch", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_json = (repo_root / args.output_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    crispai_script = repo_root / "data" / "public_benchmarks" / "sources" / "crispAI_crispr-offtarget-uncertainty" / "crispAI_score" / "crispAI.py"
    compat_script = repo_root / "scripts" / "apply_crispai_upstream_compat.py"

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"

    subprocess.run(
        [args.python_bin, str(compat_script), "--repo-root", str(repo_root)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    cmd = [
        args.python_bin,
        str(crispai_script),
        "--mode",
        "offt-score",
        "--input_file",
        str((repo_root / args.input_file).resolve()),
        "--N_samples",
        str(args.n_samples),
        "--N_mismatch",
        str(args.n_mismatch),
        "--O",
        str((repo_root / args.output_csv).resolve()),
        "--gpu",
        "-1",
    ]

    proc = subprocess.run(
        cmd,
        cwd=crispai_script.parent,
        env=env,
        capture_output=True,
        text=True,
    )

    status = "passed" if proc.returncode == 0 else "blocked"
    payload = {
        "model": "crispAI_upstream_offt_score_smoke",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_bin": args.python_bin,
        "command": cmd,
        "returncode": int(proc.returncode),
        "status": status,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-80:]),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
