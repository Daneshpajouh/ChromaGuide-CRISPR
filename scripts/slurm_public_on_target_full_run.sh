#!/bin/bash
#SBATCH --job-name=pub_on_full
#SBATCH --account=def-kwiese_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/public_on_target_full_%j.out
#SBATCH --error=slurm_logs/public_on_target_full_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/chromaguide_experiments}"
VENV_DIR="${VENV_DIR:-$HOME/env_public_benchmark}"
DEVICE="${DEVICE:-cuda}"
JOB_ID="${SLURM_JOB_ID:-local}"
RUN_TAG="${RUN_TAG:-public_on_target_full_${JOB_ID}}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
D_MODEL="${D_MODEL:-96}"
ENCODER_TYPE="${ENCODER_TYPE:-mamba}"
FUSION="${FUSION:-gate}"
LOSS_TYPE="${LOSS_TYPE:-beta}"
PRETRAIN="${PRETRAIN:-none}"
LR="${LR:-0.0006251028636335222}"
DROPOUT="${DROPOUT:-0.10617534828874074}"
FOLDS="${FOLDS:-5}"
CV_OUTPUT_ROOT="${CV_OUTPUT_ROOT:-results/public_benchmarks/${RUN_TAG}/cv}"
CV_MANIFEST="${CV_MANIFEST:-results/public_benchmarks/${RUN_TAG}/cv_manifest.json}"
TRANSFER_OUTPUT_ROOT="${TRANSFER_OUTPUT_ROOT:-results/public_benchmarks/${RUN_TAG}/transfer}"
TRANSFER_MANIFEST="${TRANSFER_MANIFEST:-results/public_benchmarks/${RUN_TAG}/transfer_manifest.json}"
FINAL_SUMMARY="${FINAL_SUMMARY:-results/public_benchmarks/${RUN_TAG}/FINAL_SUMMARY.json}"
DRY_RUN="${DRY_RUN:-0}"
VENV_BOOTSTRAP="${VENV_BOOTSTRAP:-0}"

mkdir -p "$REPO_DIR/slurm_logs"
cd "$REPO_DIR"

CV_CMD=(
  python
  scripts/run_public_on_target_benchmark.py
  --repo-root .
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --d-model "$D_MODEL"
  --device "$DEVICE"
  --encoder-type "$ENCODER_TYPE"
  --fusion "$FUSION"
  --loss-type "$LOSS_TYPE"
  --lr "$LR"
  --dropout "$DROPOUT"
  --pretrain "$PRETRAIN"
  --folds "$FOLDS"
  --output-root "$CV_OUTPUT_ROOT"
  --summary-json "$CV_MANIFEST"
)

TRANSFER_CMD=(
  python
  scripts/run_public_on_target_transfer_benchmark.py
  --repo-root .
  --source-dataset WT
  --target-dataset HL60
  --folds "$FOLDS"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --d-model "$D_MODEL"
  --device "$DEVICE"
  --encoder-type "$ENCODER_TYPE"
  --fusion "$FUSION"
  --loss-type "$LOSS_TYPE"
  --lr "$LR"
  --dropout "$DROPOUT"
  --pretrain "$PRETRAIN"
  --output-root "$TRANSFER_OUTPUT_ROOT"
  --summary-json "$TRANSFER_MANIFEST"
)

SUMMARY_CMD=(
  python
  scripts/summarize_public_on_target_full.py
  --repo-root .
  --input-dir "$CV_OUTPUT_ROOT"
  --output "$FINAL_SUMMARY"
)

if [ "$DRY_RUN" = "1" ]; then
  unset PYTHONPATH || true
  export PYTHONNOUSERSITE=1
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
  echo "DRY_RUN: ${CV_CMD[*]}"
  echo "DRY_RUN: ${TRANSFER_CMD[*]}"
  echo "DRY_RUN: ${SUMMARY_CMD[*]}"
  exit 0
fi

if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
  # Some clusters provide `module` as a shell function via profile scripts.
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

echo "Running: ${CV_CMD[*]}"
"${CV_CMD[@]}"
echo "Running: ${TRANSFER_CMD[*]}"
"${TRANSFER_CMD[@]}"
echo "Running: ${SUMMARY_CMD[*]}"
"${SUMMARY_CMD[@]}"
