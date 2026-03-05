#!/bin/bash
#SBATCH --job-name=pub_off_unc_tri
#SBATCH --account=def-kwiese
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --output=slurm_logs/public_off_target_uncertainty_trillium_%j.out
#SBATCH --error=slurm_logs/public_off_target_uncertainty_trillium_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/amird/chromaguide_experiments}"
VENV_DIR="${VENV_DIR:-/scratch/amird/env_public_benchmark}"
MAX_ROWS="${MAX_ROWS:-0}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-0.0003}"
OUTPUT_JSON="${OUTPUT_JSON:-results/public_benchmarks/public_off_target_uncertainty_change_seq_trillium.json}"
MODEL_OUT="${MODEL_OUT:-results/public_benchmarks/public_off_target_uncertainty_change_seq_trillium.pt}"
DRY_RUN="${DRY_RUN:-0}"
VENV_BOOTSTRAP="${VENV_BOOTSTRAP:-0}"

mkdir -p "$REPO_DIR/slurm_logs"
cd "$REPO_DIR"

CMD=(
  python
  scripts/train_public_off_target_uncertainty.py
  --device cpu
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

python scripts/stage_change_seq_proxy_table.py
python scripts/build_public_off_target_frames.py
echo "Running: ${CMD[*]}"
"${CMD[@]}"
