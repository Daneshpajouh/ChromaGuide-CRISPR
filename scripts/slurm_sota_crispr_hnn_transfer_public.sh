#!/bin/bash
#SBATCH --job-name=sota_hnn_xfer
#SBATCH --account=def-kwiese_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/sota_crispr_hnn_transfer_%j.out
#SBATCH --error=slurm_logs/sota_crispr_hnn_transfer_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/amird/chromaguide_experiments}"
VENV_DIR="${VENV_DIR:-/scratch/amird/env_public_benchmark_hnn}"
SOURCE_DATASET="${SOURCE_DATASET:-WT}"
TARGET_DATASET="${TARGET_DATASET:-HL60}"
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
PATIENCE="${PATIENCE:-10}"
MAX_ROWS="${MAX_ROWS:-0}"
SEED="${SEED:-2024}"
RUN_TAG="${RUN_TAG:-sota_crispr_hnn_transfer_${SLURM_JOB_ID:-local}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/public_benchmarks/${RUN_TAG}}"
SUMMARY_JSON="${SUMMARY_JSON:-results/public_benchmarks/${RUN_TAG}/SUMMARY.json}"
VENV_BOOTSTRAP="${VENV_BOOTSTRAP:-0}"

mkdir -p "$REPO_DIR/slurm_logs"
cd "$REPO_DIR"

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

if ! python - <<'PY' >/dev/null 2>&1
import tensorflow as tf
print(tf.__version__)
PY
then
  python -m pip install 'tensorflow>=2.16,<2.18' >/dev/null
fi

if ! python - <<'PY' >/dev/null 2>&1
import keras_multi_head
print(keras_multi_head.__version__)
PY
then
  python -m pip install 'keras-multi-head==0.29.0' >/dev/null
fi

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

CMD=(
  python
  scripts/run_sota_crispr_hnn_transfer_public.py
  --repo-root .
  --source-dataset "$SOURCE_DATASET"
  --target-dataset "$TARGET_DATASET"
  --folds "$FOLDS"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --patience "$PATIENCE"
  --max-rows "$MAX_ROWS"
  --seed "$SEED"
  --output-root "$OUTPUT_ROOT"
  --summary-json "$SUMMARY_JSON"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
