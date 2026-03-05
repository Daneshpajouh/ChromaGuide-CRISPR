#!/usr/bin/env python3
"""Optuna tuner for off-target focal-loss training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import optuna


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna tuning for train_off_target_focal.py")
    p.add_argument("--study-name", default="offtarget_optuna")
    p.add_argument(
        "--storage",
        default="sqlite:///results/runs/optuna/offtarget_optuna.db",
        help="Optuna storage URL, e.g. sqlite:////scratch/.../study.db",
    )
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--timeout-sec", type=int, default=0)
    p.add_argument("--sampler-seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="results/runs")
    p.add_argument("--base-prefix", default="optuna_offtarget")
    p.add_argument("--train-script", default="scripts/train_off_target_focal.py")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument(
        "--data-path",
        default="data/raw/crisprofft/CRISPRoffT_all_targets.txt",
    )
    p.add_argument("--summary-json", default="")
    p.add_argument("--queue-best-default", action="store_true")
    return p.parse_args()


def build_objective(args: argparse.Namespace):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        epochs = trial.suggest_int("epochs", 80, 220, step=20)
        batch_size = trial.suggest_categorical("batch_size", [256, 384, 512, 768])
        lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
        base_channels = trial.suggest_categorical("base_channels", [128, 192, 256, 320])
        fc_hidden = trial.suggest_categorical("fc_hidden", [128, 192, 256, 320])
        conv_dropout = trial.suggest_float("conv_dropout", 0.15, 0.50)
        fc_dropout = trial.suggest_float("fc_dropout", 0.10, 0.45)
        focal_alpha = trial.suggest_float("focal_alpha", 0.10, 0.45)
        focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.5)
        scheduler_factor = trial.suggest_float("scheduler_factor", 0.3, 0.7)
        scheduler_patience = trial.suggest_int("scheduler_patience", 3, 8)
        early_stop_patience = trial.suggest_int("early_stop_patience", 10, 24)
        max_samples = trial.suggest_categorical("max_samples", [0, 120000, 180000, 240000])
        seed = trial.suggest_int("seed", 1, 99999)

        tag = f"{args.base_prefix}_t{trial.number:04d}"
        metrics = out_dir / f"{tag}.json"
        model_out = out_dir / f"{tag}.pt"
        stdout_log = out_dir / f"{tag}.out.log"
        stderr_log = out_dir / f"{tag}.err.log"

        cmd = [
            args.python_bin,
            args.train_script,
            "--data_path",
            args.data_path,
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
            "--lr",
            f"{lr:.8g}",
            "--weight_decay",
            f"{weight_decay:.8g}",
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
            "--scheduler_factor",
            f"{scheduler_factor:.8g}",
            "--scheduler_patience",
            str(scheduler_patience),
            "--early_stop_patience",
            str(early_stop_patience),
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--output_json",
            str(metrics),
            "--model_out",
            str(model_out),
        ]
        if max_samples > 0:
            cmd.extend(["--max_samples", str(max_samples)])

        t0 = time.time()
        with stdout_log.open("w") as so, stderr_log.open("w") as se:
            proc = subprocess.run(cmd, stdout=so, stderr=se, check=False)
        elapsed = time.time() - t0

        if proc.returncode != 0 or not metrics.exists():
            trial.set_user_attr("status", "failed")
            trial.set_user_attr("returncode", proc.returncode)
            trial.set_user_attr("metrics_file", str(metrics))
            trial.set_user_attr("stdout_log", str(stdout_log))
            trial.set_user_attr("stderr_log", str(stderr_log))
            trial.set_user_attr("elapsed_sec", elapsed)
            return -1.0

        try:
            payload = json.loads(metrics.read_text())
            auroc = float(payload.get("best_auroc", -1.0))
            auprc = float(payload.get("best_auprc", -1.0))
        except Exception:
            auroc = -1.0
            auprc = -1.0

        trial.set_user_attr("status", "ok")
        trial.set_user_attr("metrics_file", str(metrics))
        trial.set_user_attr("model_out", str(model_out))
        trial.set_user_attr("stdout_log", str(stdout_log))
        trial.set_user_attr("stderr_log", str(stderr_log))
        trial.set_user_attr("elapsed_sec", elapsed)
        trial.set_user_attr("best_auprc", auprc)
        return auroc

    return objective


def create_study_with_retry(args: argparse.Namespace) -> optuna.Study:
    """Retry study initialization to avoid SQLite schema bootstrap races."""
    last_err: Exception | None = None
    for attempt in range(10):
        try:
            storage = optuna.storages.RDBStorage(
                url=args.storage,
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


def main() -> None:
    args = parse_args()
    study = create_study_with_retry(args)

    if args.queue_best_default:
        study.enqueue_trial(
            {
                "epochs": 200,
                "batch_size": 512,
                "lr": 5e-4,
                "weight_decay": 1e-5,
                "base_channels": 256,
                "fc_hidden": 256,
                "conv_dropout": 0.4,
                "fc_dropout": 0.3,
                "focal_alpha": 0.25,
                "focal_gamma": 2.0,
                "scheduler_factor": 0.5,
                "scheduler_patience": 5,
                "early_stop_patience": 15,
                "max_samples": 0,
                "seed": 42,
            }
        )

    timeout = None if args.timeout_sec <= 0 else args.timeout_sec
    study.optimize(
        build_objective(args),
        n_trials=args.n_trials,
        timeout=timeout,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    best = study.best_trial
    out_summary = (
        Path(args.summary_json)
        if args.summary_json
        else Path(args.output_dir) / f"{args.base_prefix}_study_summary.json"
    )
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(
        json.dumps(
            {
                "study_name": args.study_name,
                "storage": args.storage,
                "n_trials_total": len(study.trials),
                "best_trial_number": best.number,
                "best_value": float(best.value),
                "best_params": best.params,
                "best_user_attrs": best.user_attrs,
            },
            indent=2,
        )
    )

    print(f"Study: {args.study_name}")
    print(f"Trials: {len(study.trials)}")
    print(f"Best AUROC: {best.value:.10f}")
    print(f"Best trial: {best.number}")
    print(f"Summary: {out_summary}")


if __name__ == "__main__":
    main()
