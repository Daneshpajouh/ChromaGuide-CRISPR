#!/bin/bash
#SBATCH --job-name=pub_off_cclmoff_cpu
#SBATCH --account=def-kwiese
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/public_off_target_cclmoff_cpu_%j.out
#SBATCH --error=slurm_logs/public_off_target_cclmoff_cpu_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/amird/chromaguide_experiments}"
VENV_DIR="${VENV_DIR:-/scratch/amird/env_public_benchmark}"
DEVICE="${DEVICE:-cpu}"
METHODS="${METHODS:-CIRCLE-seq,__BLANK__}"
MAX_ROWS="${MAX_ROWS:-200000}"
NEGATIVE_KEEP_PROB="${NEGATIVE_KEEP_PROB:-0.01}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"
JOB_ID="${SLURM_JOB_ID:-local}"
OUTPUT_JSON="${OUTPUT_JSON:-results/public_benchmarks/public_off_target_cclmoff_cpu_${JOB_ID}.json}"
MODEL_OUT="${MODEL_OUT:-results/public_benchmarks/public_off_target_cclmoff_cpu_${JOB_ID}.pt}"
SPLIT_MODE="${SPLIT_MODE:-guide_holdout}"
FOLD_COUNT="${FOLD_COUNT:-5}"
FOLD_INDEX="${FOLD_INDEX:--1}"
MANIFEST_JSON="${MANIFEST_JSON:-}"
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
    scripts/train_public_off_target_cclmoff.py
    --device "$DEVICE"
    --methods "$METHODS"
    --max_rows "$MAX_ROWS"
    --negative_keep_prob "$NEGATIVE_KEEP_PROB"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --split-mode "$SPLIT_MODE"
    --fold-count "$FOLD_COUNT"
    --fold-index "$FOLD_INDEX"
    --output_json "$OUTPUT_JSON"
    --model_out "$MODEL_OUT"
  )
  if [ -n "$MANIFEST_JSON" ]; then
    CMD+=(--manifest-json "$MANIFEST_JSON" --split-mode manifest)
  fi
  echo "DRY_RUN: ${CMD[*]}"
  exit 0
fi

if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load python/3.11 || true
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
  scripts/train_public_off_target_cclmoff.py
  --device "$DEVICE"
  --methods "$METHODS"
  --max_rows "$MAX_ROWS"
  --negative_keep_prob "$NEGATIVE_KEEP_PROB"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --split-mode "$SPLIT_MODE"
  --fold-count "$FOLD_COUNT"
  --fold-index "$FOLD_INDEX"
  --output_json "$OUTPUT_JSON"
  --model_out "$MODEL_OUT"
)

if [ -n "$MANIFEST_JSON" ]; then
  CMD+=(--manifest-json "$MANIFEST_JSON" --split-mode manifest)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
