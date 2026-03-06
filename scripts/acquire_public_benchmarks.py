#!/usr/bin/env python3
"""Acquire immediately accessible public benchmark sources.

This script prioritizes sources we can fetch reproducibly today:
- git repositories that host benchmark-ready or mirror data,
- lightweight metadata pointers,
- a local source map for later preprocessing.

It intentionally does not auto-download large journal supplements or opaque Figshare bundles.
Those remain explicit manual steps in the acquisition checklist.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


REPO_SOURCES = {
    "CRISPR_HNN": "https://github.com/xx0220/CRISPR_HNN.git",
    "DeepHF": "https://github.com/izhangcd/DeepHF.git",
    "dagrate_public_data_crisprCas9": "https://github.com/dagrate/public_data_crisprCas9.git",
    "crisporPaper": "https://github.com/maximilianh/crisporPaper.git",
    "guideseq": "https://github.com/tsailabSJ/guideseq.git",
    "circleseq": "https://github.com/tsailabSJ/circleseq.git",
    "changeseq": "https://github.com/tsailabSJ/changeseq.git",
}


def resolve_git_bin() -> str:
    for candidate in [
        shutil.which("git"),
        "/usr/bin/git",
        "/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/git",
        "/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v4/usr/bin/git",
    ]:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("git executable not found in PATH or known cluster locations")


def clone_or_update(name: str, url: str, dest: Path, git_bin: str) -> dict:
    if dest.exists():
        return {
            "name": name,
            "url": url,
            "path": str(dest),
            "status": "already_present",
        }
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [git_bin, "clone", "--depth", "1", url, str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "name": name,
        "url": url,
        "path": str(dest),
        "status": "cloned" if proc.returncode == 0 else "failed",
        "stderr": proc.stderr.strip() or None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Acquire immediately accessible public benchmark sources.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument(
        "--output-json",
        default="data/public_benchmarks/acquisition/source_fetch_status.json",
        help="Write clone status here.",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    sources_root = repo / "data" / "public_benchmarks" / "sources"
    fetch_results = []
    git_bin = resolve_git_bin()

    for name, url in REPO_SOURCES.items():
        dest = sources_root / name
        fetch_results.append(clone_or_update(name, url, dest, git_bin))

    source_map = {
        "generated": "2026-03-02",
        "sources_root": str(sources_root),
        "repos": {item["name"]: item["path"] for item in fetch_results if item["status"] in {"cloned", "already_present"}},
        "manual_downloads_still_required": {
            "cclmoff_figshare": "https://doi.org/10.6084/m9.figshare.27080566.v2",
            "dnabert_epi_supplement": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12611124/",
            "deep_spcas9_supplement": "https://doi.org/10.1126/sciadv.aax9249",
            "crispron_supplement": "https://doi.org/10.1038/s41467-021-23576-0"
        }
    }

    out_path = repo / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"fetch_results": fetch_results, "source_map": source_map}, indent=2), encoding="utf-8")
    source_map_path = repo / "data" / "public_benchmarks" / "acquisition" / "source_map.json"
    source_map_path.parent.mkdir(parents=True, exist_ok=True)
    source_map_path.write_text(json.dumps(source_map, indent=2), encoding="utf-8")
    print(json.dumps({"fetch_results": fetch_results, "source_map": source_map}, indent=2))


if __name__ == "__main__":
    main()
