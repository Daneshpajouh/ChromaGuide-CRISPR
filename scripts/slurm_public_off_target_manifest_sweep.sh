#!/bin/bash
#SBATCH --job-name=pub_off_sweep
#SBATCH --account=def-kwiese_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/public_off_target_manifest_sweep_%j.out
#SBATCH --error=slurm_logs/public_off_target_manifest_sweep_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/chromaguide_experiments}"
VENV_DIR="${VENV_DIR:-$HOME/env_public_benchmark}"
DEVICE="${DEVICE:-cuda}"
MANIFEST_JSON="${MANIFEST_JSON:-data/public_benchmarks/off_target/frames/cclmoff_lodo.json}"
JOB_ID="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="${OUTPUT_DIR:-results/public_benchmarks/off_target_manifest_sweep_${JOB_ID}}"
SUMMARY_JSON="${SUMMARY_JSON:-results/public_benchmarks/off_target_manifest_sweep_${JOB_ID}/SUMMARY.json}"
MAX_ROWS="${MAX_ROWS:-200000}"
NEGATIVE_KEEP_PROB="${NEGATIVE_KEEP_PROB:-0.01}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-0.0005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
BASE_CHANNELS="${BASE_CHANNELS:-256}"
FC_HIDDEN="${FC_HIDDEN:-256}"
CONV_DROPOUT="${CONV_DROPOUT:-0.4}"
FC_DROPOUT="${FC_DROPOUT:-0.3}"
FOCAL_ALPHA="${FOCAL_ALPHA:-0.25}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
VENV_BOOTSTRAP="${VENV_BOOTSTRAP:-0}"

mkdir -p "$REPO_DIR/slurm_logs"
cd "$REPO_DIR"

CMD=(
  python
  scripts/run_public_off_target_manifest_sweep.py
  --repo-root .
  --python-bin "$VENV_DIR/bin/python"
  --manifest-json "$MANIFEST_JSON"
  --output-dir "$OUTPUT_DIR"
  --summary-json "$SUMMARY_JSON"
  --device "$DEVICE"
  --max-rows "$MAX_ROWS"
  --negative-keep-prob "$NEGATIVE_KEEP_PROB"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --base-channels "$BASE_CHANNELS"
  --fc-hidden "$FC_HIDDEN"
  --conv-dropout "$CONV_DROPOUT"
  --fc-dropout "$FC_DROPOUT"
  --focal-alpha "$FOCAL_ALPHA"
  --focal-gamma "$FOCAL_GAMMA"
  --seed "$SEED"
)

if [ "$DRY_RUN" = "1" ]; then
  unset PYTHONPATH || true
  export PYTHONNOUSERSITE=1
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
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

echo "Running: ${CMD[*]}"
"${CMD[@]}"
