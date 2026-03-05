#!/usr/bin/env python3
"""Optuna tuner for split-A non-stacked on-target models."""

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
    p = argparse.ArgumentParser(description="Optuna tuning for train_on_target_trainval.py")
    p.add_argument("--split", choices=["A", "B", "C"], default="A")
    p.add_argument("--study-name", default="splitA_optuna")
    p.add_argument(
        "--storage",
        default="sqlite:///results/runs/optuna/splitA_optuna.db",
        help="Optuna storage URL, e.g. sqlite:////scratch/.../study.db",
    )
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--timeout-sec", type=int, default=0)
    p.add_argument("--sampler-seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="results/runs")
    p.add_argument("--base-prefix", default="optuna_splitA")
    p.add_argument("--train-script", default="scripts/train_on_target_trainval.py")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--queue-best-default", action="store_true")
    p.add_argument("--summary-json", default="")
    return p.parse_args()


def build_objective(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        d_model = trial.suggest_categorical("d_model", [768, 896, 1024, 1152])
        fusion = trial.suggest_categorical("fusion", ["cross_attention", "gate"])
        loss_type = trial.suggest_categorical("loss_type", ["beta", "mse"])
        finetune_lr = trial.suggest_float("finetune_lr", 1.2e-4, 1.8e-4, log=True)
        finetune_epochs = trial.suggest_int("finetune_epochs", 160, 220, step=20)
        patience = trial.suggest_int("patience", 40, 60, step=2)
        pretrain_mix = trial.suggest_float("finetune_pretrain_mix", 0.20, 0.35)
        pretrain_decay = trial.suggest_float("finetune_pretrain_mix_decay", 0.95, 0.99)
        use_cellline = trial.suggest_categorical("use_cellline_feature", [True, False])
        seed = trial.suggest_int("seed", 10000, 99999)

        trial_tag = f"{args.base_prefix}_t{trial.number:04d}"
        output_prefix = output_dir / trial_tag
        stdout_log = output_dir / f"{trial_tag}.out.log"
        stderr_log = output_dir / f"{trial_tag}.err.log"

        cmd = [
            args.python_bin,
            args.train_script,
            "--split",
            args.split,
            "--device",
            args.device,
            "--encoder_type",
            "cnn_gru",
            "--d_model",
            str(d_model),
            "--fusion",
            fusion,
            "--batch_size",
            "64",
            "--pretrain",
            "hfesp",
            "--pretrain_lr",
            "3e-4",
            "--loss_type",
            loss_type,
            "--patience",
            str(patience),
            "--seed",
            str(seed),
            "--finetune_lr",
            f"{finetune_lr:.8g}",
            "--finetune_epochs",
            str(finetune_epochs),
            "--finetune_pretrain_mix",
            f"{pretrain_mix:.8g}",
            "--finetune_pretrain_mix_decay",
            f"{pretrain_decay:.8g}",
            "--output_prefix",
            str(output_prefix),
        ]
        if use_cellline:
            cmd.append("--use_cellline_feature")
        else:
            cmd.append("--no_cellline_feature")

        t0 = time.time()
        with stdout_log.open("w") as so, stderr_log.open("w") as se:
            proc = subprocess.run(cmd, stdout=so, stderr=se, check=False)
        elapsed = time.time() - t0

        metrics_file = Path(f"{output_prefix}.json")
        if proc.returncode != 0 or not metrics_file.exists():
            trial.set_user_attr("status", "failed")
            trial.set_user_attr("returncode", proc.returncode)
            trial.set_user_attr("metrics_file", str(metrics_file))
            trial.set_user_attr("stdout_log", str(stdout_log))
            trial.set_user_attr("stderr_log", str(stderr_log))
            trial.set_user_attr("elapsed_sec", elapsed)
            return -1.0

        try:
            payload = json.loads(metrics_file.read_text())
            rho = float(payload.get("gold_rho", -1.0))
        except Exception:
            rho = -1.0

        trial.set_user_attr("status", "ok")
        trial.set_user_attr("metrics_file", str(metrics_file))
        trial.set_user_attr("stdout_log", str(stdout_log))
        trial.set_user_attr("stderr_log", str(stderr_log))
        trial.set_user_attr("elapsed_sec", elapsed)
        return rho

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
                "d_model": 896,
                "fusion": "cross_attention",
                "loss_type": "beta",
                "finetune_lr": 1.5e-4,
                "finetune_epochs": 180,
                "patience": 46,
                "finetune_pretrain_mix": 0.28,
                "finetune_pretrain_mix_decay": 0.97,
                "use_cellline_feature": True,
                "seed": 11602,
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

    out_summary = (
        Path(args.summary_json)
        if args.summary_json
        else Path(args.output_dir) / f"{args.base_prefix}_study_summary.json"
    )
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    best = study.best_trial
    summary = {
        "study_name": args.study_name,
        "storage": args.storage,
        "n_trials_total": len(study.trials),
        "best_trial_number": best.number,
        "best_value": float(best.value),
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
    }
    out_summary.write_text(json.dumps(summary, indent=2))

    print(f"Study: {args.study_name}")
    print(f"Trials: {len(study.trials)}")
    print(f"Best rho: {best.value:.10f}")
    print(f"Best trial: {best.number}")
    print(f"Summary: {out_summary}")


if __name__ == "__main__":
    main()
