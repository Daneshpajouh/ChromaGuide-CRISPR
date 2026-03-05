#!/usr/bin/env python3
"""Optuna tuner for the public on-target benchmark using a threshold-driving scout set.

This is intentionally a scout tuner, not a full 9-dataset claim run.
Each trial evaluates a constrained architecture/hyperparameter configuration on:
- WT
- ESP
- HF
- Sniper-Cas9
- HL60
and optionally a matched WT->HL60 transfer fold.
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


SCOUT_DATASETS = ["WT", "ESP", "HF", "Sniper-Cas9", "HL60"]


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna tuning for the public on-target scout benchmark")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--study-name", default="public_on_target_scout")
    p.add_argument(
        "--storage",
        default="sqlite:///results/public_benchmarks/optuna/public_on_target_scout.db",
        help="Optuna storage URL",
    )
    p.add_argument("--n-trials", type=int, default=8)
    p.add_argument("--timeout-sec", type=int, default=0)
    p.add_argument("--sampler-seed", type=int, default=42)
    p.add_argument("--device", default="mps")
    p.add_argument("--folds", type=int, default=1, help="Scout folds per dataset")
    p.add_argument("--include-transfer", action="store_true")
    p.add_argument("--transfer-folds", type=int, default=1)
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--output-dir", default="results/public_benchmarks/optuna/on_target")
    p.add_argument("--base-prefix", default="public_on_target_optuna")
    p.add_argument("--summary-json", default="")
    return p.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _safe_remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            child.rmdir()
    if path.exists():
        path.rmdir()


def create_study_with_retry(args: argparse.Namespace) -> optuna.Study:
    storage_url = args.storage
    if storage_url.startswith("sqlite:///"):
        db_path = Path(storage_url.removeprefix("sqlite:///"))
        if not db_path.is_absolute():
            db_path = Path(args.repo_root).resolve() / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{db_path}"

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


def build_objective(args: argparse.Namespace):
    repo = Path(args.repo_root).resolve()
    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = _load_json(repo / "public_claim_thresholds.json")

    def objective(trial: optuna.Trial) -> float:
        encoder_type = trial.suggest_categorical("encoder_type", ["cnn_gru", "mamba"])
        d_model = trial.suggest_categorical("d_model", [96, 128, 160])
        fusion = trial.suggest_categorical("fusion", ["gate", "cross_attention"])
        loss_type = trial.suggest_categorical("loss_type", ["beta", "mse"])
        lr = trial.suggest_float("lr", 2e-4, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.10, 0.40)
        batch_size = trial.suggest_categorical("batch_size", [256, 512])
        epochs = trial.suggest_categorical("epochs", [10, 12, 16])

        tag = f"{args.base_prefix}_t{trial.number:04d}"
        cv_dir = output_dir / tag / "cv"
        transfer_dir = output_dir / tag / "transfer"
        cv_summary = output_dir / tag / "cv_summary.json"
        stdout_log = output_dir / f"{tag}.out.log"
        stderr_log = output_dir / f"{tag}.err.log"

        cv_cmd = [
            args.python_bin,
            "scripts/run_public_on_target_benchmark.py",
            "--repo-root",
            str(repo),
            "--datasets",
            *SCOUT_DATASETS,
            "--folds",
            str(args.folds),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--d-model",
            str(d_model),
            "--device",
            args.device,
            "--encoder-type",
            encoder_type,
            "--fusion",
            fusion,
            "--loss-type",
            loss_type,
            "--lr",
            f"{lr:.8g}",
            "--dropout",
            f"{dropout:.8g}",
            "--output-root",
            str(cv_dir),
            "--summary-json",
            str(output_dir / tag / "cv_manifest.json"),
        ]

        transfer_cmd = [
            args.python_bin,
            "scripts/run_public_on_target_transfer_benchmark.py",
            "--repo-root",
            str(repo),
            "--source-dataset",
            "WT",
            "--target-dataset",
            "HL60",
            "--folds",
            str(args.transfer_folds),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--d-model",
            str(d_model),
            "--device",
            args.device,
            "--encoder-type",
            encoder_type,
            "--fusion",
            fusion,
            "--loss-type",
            loss_type,
            "--lr",
            f"{lr:.8g}",
            "--dropout",
            f"{dropout:.8g}",
            "--output-root",
            str(transfer_dir),
            "--summary-json",
            str(output_dir / tag / "transfer_manifest.json"),
        ]

        t0 = time.time()
        rc = 0
        try:
            with stdout_log.open("w") as so, stderr_log.open("w") as se:
                rc = subprocess.run(
                    cv_cmd,
                    cwd=repo,
                    stdout=so,
                    stderr=se,
                    check=False,
                    env=_clean_env(),
                ).returncode
                if rc == 0:
                    sum_cmd = [
                        sys.executable,
                        "scripts/summarize_public_on_target_full.py",
                        "--repo-root",
                        str(repo),
                        "--input-dir",
                        str(cv_dir.relative_to(repo)),
                        "--output",
                        str(cv_summary),
                    ]
                    rc = subprocess.run(
                        sum_cmd,
                        cwd=repo,
                        stdout=so,
                        stderr=se,
                        check=False,
                        env=_clean_env(),
                    ).returncode
                if rc == 0 and args.include_transfer:
                    rc = subprocess.run(
                        transfer_cmd,
                        cwd=repo,
                        stdout=so,
                        stderr=se,
                        check=False,
                        env=_clean_env(),
                    ).returncode
        except Exception:
            rc = 1
        elapsed = time.time() - t0

        if rc != 0 or not cv_summary.exists():
            trial.set_user_attr("status", "failed")
            trial.set_user_attr("returncode", rc)
            trial.set_user_attr("stdout_log", str(stdout_log))
            trial.set_user_attr("stderr_log", str(stderr_log))
            trial.set_user_attr("elapsed_sec", elapsed)
            return -1.0

        payload = _load_json(cv_summary)["summary"]
        score_terms = []
        ds_details = {}
        for ds in SCOUT_DATASETS:
            ds_metric = payload["completed_datasets"][ds]["mean_gold_rho"]
            if ds == "WT":
                threshold = thresholds["on_target"]["per_dataset_thresholds"]["WT_SCC"]
            elif ds == "ESP":
                threshold = thresholds["on_target"]["per_dataset_thresholds"]["ESP_SCC"]
            elif ds == "HF":
                threshold = thresholds["on_target"]["per_dataset_thresholds"]["HF_SCC"]
            elif ds == "Sniper-Cas9":
                threshold = thresholds["on_target"]["per_dataset_thresholds"]["Sniper_Cas9_SCC"]
            elif ds == "HL60":
                threshold = thresholds["on_target"]["per_dataset_thresholds"]["HL60_SCC"]
            else:
                threshold = 1.0
            ratio = ds_metric / threshold if threshold > 0 else ds_metric
            score_terms.append(ratio)
            ds_details[ds] = {
                "mean_gold_rho": ds_metric,
                "threshold": threshold,
                "ratio": ratio,
            }

        transfer_detail = None
        if args.include_transfer:
            vals = []
            for p in sorted(transfer_dir.glob("WT_to_HL60_fold*.json")):
                vals.append(float(_load_json(p)["gold_rho"]))
            if vals:
                mean_transfer = sum(vals) / len(vals)
                threshold = thresholds["on_target"]["per_dataset_thresholds"]["WT_to_HL60_SCC"]
                ratio = mean_transfer / threshold if threshold > 0 else mean_transfer
                score_terms.append(ratio)
                transfer_detail = {
                    "mean_gold_rho": mean_transfer,
                    "threshold": threshold,
                    "ratio": ratio,
                }

        score = sum(score_terms) / len(score_terms)
        trial.set_user_attr("status", "ok")
        trial.set_user_attr("stdout_log", str(stdout_log))
        trial.set_user_attr("stderr_log", str(stderr_log))
        trial.set_user_attr("elapsed_sec", elapsed)
        trial.set_user_attr("cv_summary", str(cv_summary))
        trial.set_user_attr("dataset_details", ds_details)
        if transfer_detail is not None:
            trial.set_user_attr("transfer_detail", transfer_detail)
        return score

    return objective


def main() -> None:
    args = parse_args()
    study = create_study_with_retry(args)
    timeout = None if args.timeout_sec <= 0 else args.timeout_sec
    study.optimize(
        build_objective(args),
        n_trials=args.n_trials,
        timeout=timeout,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    out_summary = (
        Path(args.summary_json)
        if args.summary_json
        else Path(args.repo_root).resolve() / args.output_dir / f"{args.base_prefix}_study_summary.json"
    )
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    best = study.best_trial
    ranked_trials = sorted(
        (
            {
                "number": t.number,
                "state": str(t.state),
                "value": None if t.value is None else float(t.value),
                "params": t.params,
                "user_attrs": t.user_attrs,
            }
            for t in study.trials
        ),
        key=lambda item: (-1e18 if item["value"] is None else item["value"]),
        reverse=True,
    )
    summary = {
        "study_name": args.study_name,
        "storage": args.storage,
        "n_trials_total": len(study.trials),
        "best_trial_number": best.number,
        "best_value": float(best.value),
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "top_2_trials": ranked_trials[:2],
        "trials": ranked_trials,
    }
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"Study: {args.study_name}")
    print(f"Trials: {len(study.trials)}")
    print(f"Best scout score: {best.value:.8f}")
    print(f"Best trial: {best.number}")
    print(f"Summary: {out_summary}")


if __name__ == "__main__":
    main()
