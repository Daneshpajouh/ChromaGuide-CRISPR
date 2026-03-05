#!/bin/bash
#SBATCH --job-name=sota_fmc_pub
#SBATCH --account=def-kwiese_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/sota_crispr_fmc_%j.out
#SBATCH --error=slurm_logs/sota_crispr_fmc_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/chromaguide_experiments}"
VENV_DIR="${VENV_DIR:-$HOME/env_public_benchmark}"
DEVICE="${DEVICE:-cuda}"
DATASETS="${DATASETS:-WT,ESP,HF,Sniper-Cas9,HL60}"
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
MAX_ROWS="${MAX_ROWS:-0}"
SEED="${SEED:-220}"
VENV_BOOTSTRAP="${VENV_BOOTSTRAP:-0}"
JOB_ID="${SLURM_JOB_ID:-local}"
RUN_TAG="${RUN_TAG:-sota_crispr_fmc_${JOB_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/public_benchmarks/${RUN_TAG}}"
SUMMARY_JSON="${SUMMARY_JSON:-results/public_benchmarks/${RUN_TAG}/SUMMARY.json}"

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
import fm
print('ok')
PY
then
  python -m pip install 'rna-fm>=0.2.2' >/dev/null
fi

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-$REPO_DIR/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_DIR/.cache}"
mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME"

CMD=(
  python
  scripts/run_sota_crispr_fmc_public.py
  --repo-root .
  --datasets "$DATASETS"
  --folds "$FOLDS"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --embed-batch-size "$EMBED_BATCH_SIZE"
  --max-rows "$MAX_ROWS"
  --seed "$SEED"
  --device "$DEVICE"
  --output-root "$OUTPUT_ROOT"
  --summary-json "$SUMMARY_JSON"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
