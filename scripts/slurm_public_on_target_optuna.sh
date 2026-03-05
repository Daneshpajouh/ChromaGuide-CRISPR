#!/bin/bash
#SBATCH --job-name=pub_on_optuna
#SBATCH --account=def-kwiese_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/public_on_target_optuna_%j.out
#SBATCH --error=slurm_logs/public_on_target_optuna_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/chromaguide_experiments}"
VENV_DIR="${VENV_DIR:-$HOME/env_public_benchmark}"
DEVICE="${DEVICE:-cuda}"
N_TRIALS="${N_TRIALS:-8}"
FOLDS="${FOLDS:-1}"
TRANSFER_FOLDS="${TRANSFER_FOLDS:-1}"
INCLUDE_TRANSFER="${INCLUDE_TRANSFER:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-results/public_benchmarks/optuna_public_on_target}"
STUDY_NAME="${STUDY_NAME:-}"
STORAGE_URI="${STORAGE_URI:-}"
JOB_ID="${SLURM_JOB_ID:-local}"
SUMMARY_JSON="${SUMMARY_JSON:-results/public_benchmarks/optuna_public_on_target/OPTUNA_PUBLIC_ON_TARGET_SUMMARY_SLURM_${JOB_ID}.json}"
DRY_RUN="${DRY_RUN:-0}"
VENV_BOOTSTRAP="${VENV_BOOTSTRAP:-0}"

mkdir -p "$REPO_DIR/slurm_logs"
cd "$REPO_DIR"

if [ "$DRY_RUN" = "1" ]; then
  unset PYTHONPATH || true
  export PYTHONNOUSERSITE=1
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
  CMD=(
    python
    scripts/optuna_tune_public_on_target.py
    --repo-root .
    --device "$DEVICE"
    --n-trials "$N_TRIALS"
    --folds "$FOLDS"
    --transfer-folds "$TRANSFER_FOLDS"
    --python-bin "$VENV_DIR/bin/python"
    --output-dir "$OUTPUT_DIR"
    --summary-json "$SUMMARY_JSON"
  )
  if [ -n "$STUDY_NAME" ]; then
    CMD+=(--study-name "$STUDY_NAME")
  fi
  if [ -n "$STORAGE_URI" ]; then
    CMD+=(--storage "$STORAGE_URI")
  fi
  if [ "$INCLUDE_TRANSFER" = "1" ]; then
    CMD+=(--include-transfer)
  fi
  echo "DRY_RUN: ${CMD[*]}"
  exit 0
fi

if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load cuda/12.2 python/3.11 || true
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  if [ -d "$VENV_DIR" ]; then
    python -m venv --clear "$VENV_DIR"
  else
    python -m venv "$VENV_DIR"
  fi
fi
source "$VENV_DIR/bin/activate"
if [ "$VENV_BOOTSTRAP" = "1" ]; then
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -r requirements-public-benchmark.txt >/dev/null
fi

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

CMD=(
  python
  scripts/optuna_tune_public_on_target.py
  --repo-root .
  --device "$DEVICE"
  --n-trials "$N_TRIALS"
  --folds "$FOLDS"
  --transfer-folds "$TRANSFER_FOLDS"
  --python-bin "$VENV_DIR/bin/python"
  --output-dir "$OUTPUT_DIR"
  --summary-json "$SUMMARY_JSON"
)

if [ -n "$STUDY_NAME" ]; then
  CMD+=(--study-name "$STUDY_NAME")
fi
if [ -n "$STORAGE_URI" ]; then
  CMD+=(--storage "$STORAGE_URI")
fi
if [ "$INCLUDE_TRANSFER" = "1" ]; then
  CMD+=(--include-transfer)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
