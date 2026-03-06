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
    utils_py = crispai_dir / "utils.py"

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

    utils_text = utils_py.read_text(encoding="utf-8")
    coerce_helper = """

def _coerce_feature_vec(raw, start: int = 73, stop: int = 96) -> np.ndarray:
    \"\"\"Convert upstream comma-separated physical features into a stable 23-vector.\"\"\"
    if raw is None:
        return np.zeros((stop - start,), dtype=np.float32)
    if not isinstance(raw, str):
        raw = str(raw)
    parts = raw.split(',')[start:stop]
    if len(parts) != (stop - start):
        return np.zeros((stop - start,), dtype=np.float32)
    cleaned = []
    for val in parts:
        if val in {\"NA\", \"N/A\", \"\", \"None\", \"nan\"}:
            cleaned.append(0.0)
        else:
            try:
                cleaned.append(float(val))
            except ValueError:
                cleaned.append(0.0)
    return np.asarray(cleaned, dtype=np.float32)
"""
    combined_helper = """


def _safe_minmax_stack(values) -> tuple[list[np.ndarray], np.ndarray]:
    stacked = np.stack([np.asarray(x, dtype=np.float32) for x in values], axis=0)
    min_val = float(np.min(stacked))
    max_val = float(np.max(stacked))
    denom = max_val - min_val
    if not np.isfinite(denom) or denom <= 0:
        zeros = np.zeros_like(stacked, dtype=np.float32)
        return [row for row in zeros], zeros
    normalized = (stacked - min_val) / denom
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    return [row for row in normalized], normalized
"""
    if "def _coerce_feature_vec(" not in utils_text:
        utils_text = utils_text.replace("import pdb\n", "import pdb\n" + coerce_helper, 1)
    if "def _safe_minmax_stack(" not in utils_text:
        utils_text = replace_once(utils_text, coerce_helper, coerce_helper + combined_helper)

    legacy_block = """    nupop_nan_indexes = df[df[nupop_affinity_col].isna()].index # 7 samples in total\n    df = df[~df[nupop_affinity_col].isna()]\n    df = df.reset_index(drop=True)\n    nupop_occupancy = df[nupop_occupancy_col].apply(lambda x: np.asarray(x.split(',')[73:96], dtype=np.float32))\n    gc_flank = df[gc_content_col].apply(lambda x: np.asarray(x.split(',')[73:96], dtype=np.float32))\n    nupop_affinity = df[nupop_affinity_col].apply(lambda x: x.split(',')[73:96])\n    nucleotide_bdm = df[nucleotide_bdm_col].apply(lambda x: x.split(','))\n\n    # check if any of the nupop_affinity sequences contain 'NA' and replace with 0\n    for i in range(len(nupop_affinity)):\n        if 'NA' in nupop_affinity[i]:\n            nupop_affinity[i] = [0 if x == 'NA' else x for x in nupop_affinity[i]]\n\n    for i in range(len(nucleotide_bdm)):\n        if 'NA' in nucleotide_bdm[i]:\n            nucleotide_bdm[i] = [0 if x == 'NA' else x for x in nucleotide_bdm[i]]\n\n    nupop_affinity = nupop_affinity.apply(lambda x: np.asarray(x, dtype=np.float32))\n    nupop_occupancy = nupop_occupancy.apply(lambda x: np.asarray(x, dtype=np.float32))\n    nucleotide_bdm = nucleotide_bdm.apply(lambda x: np.asarray(x, dtype=np.float32))\n"""
    robust_block = """    df = df.reset_index(drop=True)\n    nupop_occupancy = df[nupop_occupancy_col].apply(_coerce_feature_vec)\n    gc_flank = df[gc_content_col].apply(_coerce_feature_vec)\n    nupop_affinity = df[nupop_affinity_col].apply(_coerce_feature_vec)\n    nucleotide_bdm = df[nucleotide_bdm_col].apply(_coerce_feature_vec)\n"""
    if robust_block not in utils_text:
        utils_text = replace_once(utils_text, legacy_block, robust_block)

    normalize_block = """    nupop_affinity = np.stack([x for x in df['NuPoP affinity']], axis=0)  \n    nupop_occupancy = np.stack([x for x in df['NuPoP occupancy']], axis=0)\n    gc_flank = np.stack([x for x in df['GC flank']], axis=0)\n    nucleotide_bdm = np.stack([x for x in df['nucleotide BDM']], axis=0)\n\n    df['NuPoP occupancy'] = (df['NuPoP occupancy'] - nupop_occupancy.min()) / (nupop_occupancy.max() - nupop_occupancy.min())\n    df['NuPoP affinity'] = (df['NuPoP affinity'] - nupop_affinity.min()) / (nupop_affinity.max() - nupop_affinity.min())\n    df['GC flank'] = (df['GC flank'] - gc_flank.min()) / (gc_flank.max() - gc_flank.min())\n    df['nucleotide BDM'] = (df['nucleotide BDM'] - nucleotide_bdm.min()) / (nucleotide_bdm.max() - nucleotide_bdm.min())\n"""
    safe_normalize_block = """    nupop_occupancy, _ = _safe_minmax_stack(df['NuPoP occupancy'])\n    nupop_affinity, _ = _safe_minmax_stack(df['NuPoP affinity'])\n    gc_flank, _ = _safe_minmax_stack(df['GC flank'])\n    nucleotide_bdm, _ = _safe_minmax_stack(df['nucleotide BDM'])\n\n    df['NuPoP occupancy'] = nupop_occupancy\n    df['NuPoP affinity'] = nupop_affinity\n    df['GC flank'] = gc_flank\n    df['nucleotide BDM'] = nucleotide_bdm\n"""
    if safe_normalize_block not in utils_text:
        utils_text = replace_once(utils_text, normalize_block, safe_normalize_block)

    lens_block = """    nupop_lens = df['NuPoP occupancy'].apply(lambda x: len(x))\n    nupop_lens = nupop_lens[nupop_lens != 23].index\n    df = df[~df['NuPoP occupancy'].apply(lambda x: len(x) != 23)]\n    df = df.reset_index(drop=True)\n\n"""
    if lens_block in utils_text:
        utils_text = utils_text.replace(lens_block, "", 1)
    utils_py.write_text(utils_text, encoding="utf-8")

    print(crispai_py)
    print(model_py)
    print(utils_py)


if __name__ == "__main__":
    main()
