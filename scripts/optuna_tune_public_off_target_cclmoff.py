#!/usr/bin/env python3
"""Optuna scout tuner for the direct public CCLMoff off-target trainer.

This is a pragmatic scout tuner on a filtered subset, not a claim-valid full-frame run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna tuning for the public CCLMoff off-target scout")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--study-name", default="public_off_target_cclmoff_scout")
    p.add_argument(
        "--storage",
        default="sqlite:///results/public_benchmarks/optuna/public_off_target_cclmoff_scout.db",
        help="Optuna storage URL",
    )
    p.add_argument("--n-trials", type=int, default=4)
    p.add_argument("--timeout-sec", type=int, default=0)
    p.add_argument("--sampler-seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--methods", default="CIRCLE-seq,__BLANK__")
    p.add_argument("--frame-manifest", default="")
    p.add_argument("--fold-index", type=int, default=-1)
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--output-dir", default="results/public_benchmarks/optuna_public_off_target")
    p.add_argument("--base-prefix", default="public_off_target_optuna")
    p.add_argument("--summary-json", default="")
    return p.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _resolve_storage_url(repo: Path, storage_url: str) -> str:
    if storage_url.startswith("sqlite:///"):
        db_path = Path(storage_url.removeprefix("sqlite:///"))
        if not db_path.is_absolute():
            db_path = repo / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}"
    return storage_url


def create_study_with_retry(repo: Path, args: argparse.Namespace) -> optuna.Study:
    storage_url = _resolve_storage_url(repo, args.storage)
    last_err: Exception | None = None
    for attempt in range(10):
        try:
            storage = optuna.storages.RDBStorage(
                url=storage_url,
                engine_kwargs={"connect_args": {"timeout": 120}},
            )
            sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True)
            return optuna.create_study(
                study_name=args.study_name,
                storage=storage,
                direction="maximize",
                load_if_exists=True,
                sampler=sampler,
            )
        except Exception as err:
            last_err = err
            msg = str(err)
            race = (
                "alembic_version" in msg
                or "UNIQUE constraint failed" in msg
                or "already exists" in msg
            )
            if attempt == 9 or not race:
                raise
            time.sleep(2 + attempt)
    raise RuntimeError(f"failed to initialize study: {last_err}")


def build_objective(repo: Path, args: argparse.Namespace):
    out_root = repo / args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        tag = f"{args.base_prefix}_t{trial.number:04d}"
        out_json = out_root / f"{tag}.json"
        out_model = out_root / f"{tag}.pt"
        stdout_log = out_root / f"{tag}.out.log"
        stderr_log = out_root / f"{tag}.err.log"

        lr = trial.suggest_float("lr", 2e-4, 1e-3, log=True)
        base_channels = trial.suggest_categorical("base_channels", [128, 192, 256])
        fc_hidden = trial.suggest_categorical("fc_hidden", [128, 192, 256, 320])
        conv_dropout = trial.suggest_float("conv_dropout", 0.2, 0.5)
        fc_dropout = trial.suggest_float("fc_dropout", 0.15, 0.45)
        focal_alpha = trial.suggest_float("focal_alpha", 0.15, 0.4)
        focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
        negative_keep_prob = trial.suggest_categorical("negative_keep_prob", [0.005, 0.01, 0.02])
        # Manifest-driven claim frames (especially LODO) require full method coverage.
        # Row truncation can drop whole methods and invalidate splits.
        if args.frame_manifest:
            max_rows = 0
        else:
            max_rows = trial.suggest_categorical("max_rows", [80000, 120000, 160000])
        epochs = trial.suggest_categorical("epochs", [4, 6])
        batch_size = trial.suggest_categorical("batch_size", [256, 512])

        cmd = [
            args.python_bin,
            "scripts/train_public_off_target_cclmoff.py",
            "--device",
            args.device,
            "--max_rows",
            str(max_rows),
            "--negative_keep_prob",
            str(negative_keep_prob),
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
            "--lr",
            f"{lr:.8g}",
            "--base_channels",
            str(base_channels),
            "--fc_hidden",
            str(fc_hidden),
            "--conv_dropout",
            f"{conv_dropout:.8g}",
            "--fc_dropout",
            f"{fc_dropout:.8g}",
            "--focal_alpha",
            f"{focal_alpha:.8g}",
            "--focal_gamma",
            f"{focal_gamma:.8g}",
            "--output_json",
            str(out_json),
            "--model_out",
            str(out_model),
        ]
        if args.frame_manifest:
            cmd.extend(["--manifest-json", args.frame_manifest, "--split-mode", "manifest"])
            if args.fold_index >= 0:
                cmd.extend(["--fold-index", str(args.fold_index)])
        else:
            cmd.extend(["--methods", args.methods])

        t0 = time.time()
        with stdout_log.open("w") as so, stderr_log.open("w") as se:
            rc = subprocess.run(
                cmd,
                cwd=repo,
                stdout=so,
                stderr=se,
                check=False,
                env=_clean_env(),
            ).returncode
        elapsed = time.time() - t0
        if rc != 0 or not out_json.exists():
            trial.set_user_attr("status", "failed")
            trial.set_user_attr("returncode", rc)
            trial.set_user_attr("stdout_log", str(stdout_log))
            trial.set_user_attr("stderr_log", str(stderr_log))
            trial.set_user_attr("elapsed_sec", elapsed)
            return -1.0

        payload = _load_json(out_json)
        auroc = float(payload["best_auroc"])
        auprc = float(payload["best_auprc"])
        score = 0.5 * (auroc + auprc)

        trial.set_user_attr("status", "ok")
        trial.set_user_attr("tag", tag)
        trial.set_user_attr("metrics_json", str(out_json))
        trial.set_user_attr("model_path", str(out_model))
        trial.set_user_attr("auroc", auroc)
        trial.set_user_attr("auprc", auprc)
        trial.set_user_attr("elapsed_sec", elapsed)
        return score

    return objective


def write_summary(repo: Path, study: optuna.Study, args: argparse.Namespace) -> Path:
    if args.summary_json:
        path = repo / args.summary_json
    else:
        path = repo / args.output_dir / "OPTUNA_PUBLIC_OFF_TARGET_SUMMARY.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    best = study.best_trial
    payload = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "best_trial_number": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "frame_manifest": args.frame_manifest,
        "trials": [
            {
                "number": t.number,
                "state": str(t.state),
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
            }
            for t in study.trials
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> None:
    args = parse_args()
    repo = Path(args.repo_root).resolve()
    if args.frame_manifest:
        manifest_path = Path(args.frame_manifest)
        if not manifest_path.is_absolute():
            manifest_path = repo / manifest_path
        args.frame_manifest = str(manifest_path)
    study = create_study_with_retry(repo, args)
    study.optimize(
        build_objective(repo, args),
        n_trials=args.n_trials,
        timeout=None if args.timeout_sec <= 0 else args.timeout_sec,
    )
    summary_path = write_summary(repo, study, args)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
