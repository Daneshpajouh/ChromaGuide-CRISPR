#!/bin/bash
#SBATCH --job-name=pub_off_uncert
#SBATCH --account=def-kwiese_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/public_off_target_uncertainty_%j.out
#SBATCH --error=slurm_logs/public_off_target_uncertainty_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/chromaguide_experiments}"
VENV_DIR="${VENV_DIR:-$HOME/env_public_benchmark}"
DEVICE="${DEVICE:-cuda}"
MAX_ROWS="${MAX_ROWS:-0}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-0.0003}"
OUTPUT_JSON="${OUTPUT_JSON:-results/public_benchmarks/public_off_target_uncertainty_change_seq.json}"
MODEL_OUT="${MODEL_OUT:-results/public_benchmarks/public_off_target_uncertainty_change_seq.pt}"
DRY_RUN="${DRY_RUN:-0}"
VENV_BOOTSTRAP="${VENV_BOOTSTRAP:-0}"

mkdir -p "$REPO_DIR/slurm_logs"
cd "$REPO_DIR"

CMD=(
  python
  scripts/train_public_off_target_uncertainty.py
  --device "$DEVICE"
  --max-rows "$MAX_ROWS"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --output-json "$OUTPUT_JSON"
  --model-out "$MODEL_OUT"
)

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN: python scripts/stage_change_seq_proxy_table.py"
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

python scripts/stage_change_seq_proxy_table.py
python scripts/build_public_off_target_frames.py
echo "Running: ${CMD[*]}"
"${CMD[@]}"
