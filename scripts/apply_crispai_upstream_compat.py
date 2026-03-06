#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Expected snippet not found: {old[:80]!r}")
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply local compatibility shims to the crispAI upstream checkout.")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    crispai_dir = repo_root / "data" / "public_benchmarks" / "sources" / "crispAI_crispr-offtarget-uncertainty" / "crispAI_score"
    crispai_py = crispai_dir / "crispAI.py"
    model_py = crispai_dir / "model.py"

    crispai_text = crispai_py.read_text(encoding="utf-8")
    crispai_text = replace_once(
        crispai_text,
        "checkpoint = torch.load('./model_checkpoint/epoch:19-best_valid_loss:0.270.pt')",
        "checkpoint_path = './model_checkpoint/epoch:19-best_valid_loss:0.270.pt'\nmap_location = torch.device('cpu') if args.gpu < 0 else None\ncheckpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)",
    )
    crispai_py.write_text(crispai_text, encoding="utf-8")

    model_text = model_py.read_text(encoding="utf-8")
    compat_alias = (
        "\n\n# Upstream CLI imports CrispAI and CrispAI_pi but instantiates CrispAI_pi.\n"
        "# Expose a compatibility alias so the original entrypoint can run unchanged.\n"
        "CrispAI = CrispAI_pi\n"
    )
    if compat_alias not in model_text:
        model_text = model_text.rstrip() + compat_alias
        model_py.write_text(model_text, encoding="utf-8")

    print(crispai_py)
    print(model_py)


if __name__ == "__main__":
    main()
