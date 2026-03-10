#!/usr/bin/env python3
"""Extract the exact crispAI parity artifacts from the Zenodo bundles.

This stages the published train/test CHANGE-seq tables, the best checkpoint,
and the supplementary evaluation scripts into a stable repo-local location.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


DATA_MEMBERS = {
    "train_csv": "crispAI_result_reproduction/source_data/changeseq_offtarget_data_flank73_filtered_nupop_gc_bdm_preprocessed_train.csv",
    "test_csv": "crispAI_result_reproduction/source_data/changeseq_offtarget_data_flank73_filtered_nupop_gc_bdm_preprocessed_test.csv",
    "best_checkpoint": "crispAI_result_reproduction/checkpoint/epoch:19-best_valid_loss:0.270.pt",
    "supp_tab_7": "crispAI_result_reproduction/Supplementary_tables/Supp_tab_7.py",
    "supp_tables_6_8_9_10_11_12_13_14": "crispAI_result_reproduction/Supplementary_tables/Supp_tabs_6_8_9_10_11_12_13_14.py",
    "fig4_preds": "crispAI_result_reproduction/Fig4/changeseqtest_preds_median_scores_all.npy",
    "fig4_y_test": "crispAI_result_reproduction/Fig4/y_test.npy",
}

MODEL_MEMBERS = {
    "cli_entry": "furkanozdenn-crispr-offtarget-uncertainty-cc8b72b/crispAI_score/crispAI.py",
    "model_py": "furkanozdenn-crispr-offtarget-uncertainty-cc8b72b/crispAI_score/model.py",
    "utils_py": "furkanozdenn-crispr-offtarget-uncertainty-cc8b72b/crispAI_score/utils.py",
    "negative_binomial_py": "furkanozdenn-crispr-offtarget-uncertainty-cc8b72b/crispAI_score/_negative_binomial.py",
    "encoding_py": "furkanozdenn-crispr-offtarget-uncertainty-cc8b72b/crispAI_score/encoding.py",
    "env_yml": "furkanozdenn-crispr-offtarget-uncertainty-cc8b72b/env/crispAI_env.yml",
    "r_env_csv": "furkanozdenn-crispr-offtarget-uncertainty-cc8b72b/env/R_env.csv",
}


def extract_member_from_tar(tar_path: Path, member_name: str, out_path: Path) -> dict[str, object]:
    with tarfile.open(tar_path, "r:gz") as tf:
        member = tf.getmember(member_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tf.extractfile(member) as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return {
        "source_member": member_name,
        "output_path": str(out_path),
        "size": out_path.stat().st_size,
    }


def extract_member_from_zip(zip_path: Path, member_name: str, out_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(zip_path) as zf:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member_name) as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return {
        "source_member": member_name,
        "output_path": str(out_path),
        "size": out_path.stat().st_size,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-bundle",
        default="data/external_artifact_inspection/downloads/crispAI/crispAI_result_reproduction.gz",
    )
    ap.add_argument(
        "--model-bundle",
        default="data/external_artifact_inspection/downloads/crispAI/crispAI_model.zip",
    )
    ap.add_argument(
        "--output-dir",
        default="data/public_benchmarks/off_target/crispai_parity",
    )
    ap.add_argument(
        "--output-manifest",
        default="data/public_benchmarks/off_target/crispai_parity/crispai_parity_manifest.json",
    )
    args = ap.parse_args()

    repo_root = Path.cwd().resolve()
    data_bundle = (repo_root / args.data_bundle).resolve()
    model_bundle = (repo_root / args.model_bundle).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_manifest = (repo_root / args.output_manifest).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    extracted: dict[str, dict[str, object]] = {}
    for key, member_name in DATA_MEMBERS.items():
        suffix = Path(member_name).name
        out_path = output_dir / "data_bundle" / suffix
        extracted[key] = extract_member_from_tar(data_bundle, member_name, out_path)

    for key, member_name in MODEL_MEMBERS.items():
        suffix = Path(member_name).name
        out_path = output_dir / "model_bundle" / suffix
        extracted[key] = extract_member_from_zip(model_bundle, member_name, out_path)

    supp_tab_7_text = (output_dir / "data_bundle" / "Supp_tab_7.py").read_text(encoding="utf-8", errors="replace")
    supp_tabs_text = (
        output_dir / "data_bundle" / "Supp_tabs_6_8_9_10_11_12_13_14.py"
    ).read_text(encoding="utf-8", errors="replace")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_bundle": {
            "path": str(data_bundle),
            "size": data_bundle.stat().st_size,
        },
        "model_bundle": {
            "path": str(model_bundle),
            "size": model_bundle.stat().st_size,
        },
        "output_dir": str(output_dir),
        "extracted": extracted,
        "signals": {
            "has_exact_train_csv": True,
            "has_exact_test_csv": True,
            "has_exact_best_checkpoint": True,
            "has_supplementary_test_eval_script": True,
            "supp_tab_7_uses_test_csv": "preprocessed_test.csv" in supp_tab_7_text,
            "supp_tab_7_uses_adjusted_target": "CHANGEseq_reads_adjusted" in supp_tab_7_text,
            "supp_tabs_use_test_csv": "preprocessed_test.csv" in supp_tabs_text,
            "supp_tabs_use_best_checkpoint": "epoch:19-best_valid_loss:0.270.pt" in supp_tabs_text,
            "supp_tabs_compute_spearman": "spearmanr(" in supp_tabs_text,
        },
        "verdict": "exact_artifacts_staged_for_claim_parity_execution",
        "notes": [
            "This stages the exact published crispAI train/test CHANGE-seq tables and best checkpoint from Zenodo.",
            "It does not itself prove the published 0.5114 statistic; that requires executing the matching evaluation path.",
            "The supplementary scripts reference the staged test CSV and the best checkpoint directly.",
        ],
    }

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output_manifest)


if __name__ == "__main__":
    main()
