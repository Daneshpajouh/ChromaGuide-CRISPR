#!/bin/bash
#SBATCH --job-name=sota_hnn_pub
#SBATCH --account=def-kwiese_gpu
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/sota_crispr_hnn_%j.out
#SBATCH --error=slurm_logs/sota_crispr_hnn_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/amird/chromaguide_experiments}"
VENV_DIR="${VENV_DIR:-/scratch/amird/env_public_benchmark_hnn}"
DATASETS="${DATASETS:-WT,ESP,HF,Sniper-Cas9,HL60}"
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
PATIENCE="${PATIENCE:-10}"
MAX_ROWS="${MAX_ROWS:-0}"
SEED="${SEED:-2024}"
RUN_TAG="${RUN_TAG:-sota_crispr_hnn_${SLURM_JOB_ID:-local}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/public_benchmarks/${RUN_TAG}}"
SUMMARY_JSON="${SUMMARY_JSON:-results/public_benchmarks/${RUN_TAG}/SUMMARY.json}"
VENV_BOOTSTRAP="${VENV_BOOTSTRAP:-0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.2}"

mkdir -p "$REPO_DIR/slurm_logs"
cd "$REPO_DIR"
mkdir -p "$OUTPUT_ROOT" "$(dirname "$SUMMARY_JSON")"

echo "HNN_PUBLIC_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "REPO_DIR=$REPO_DIR"
echo "VENV_DIR=$VENV_DIR"
echo "CUDA_MODULE=$CUDA_MODULE"
echo "HOSTNAME=$(hostname)"

if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
  echo "SOURCING_MODULES_SH"
  . /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  echo "LOADING_MODULES"
  module load "$CUDA_MODULE" python/3.11 || true
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "BOOTSTRAP_VENV"
  if [ -d "$VENV_DIR" ]; then
    python -m venv --clear "$VENV_DIR"
  else
    python -m venv "$VENV_DIR"
  fi
fi
echo "ACTIVATING_VENV"
source "$VENV_DIR/bin/activate"
echo "PYTHON_BIN=$(command -v python)"

run_import_check() {
  local label="$1"
  shift
  echo "${label}_CHECK_START"
  if command -v timeout >/dev/null 2>&1; then
    if timeout 120 "$@" >/dev/null 2>&1; then
      echo "${label}_CHECK_OK"
      return 0
    fi
  else
    if "$@" >/dev/null 2>&1; then
      echo "${label}_CHECK_OK"
      return 0
    fi
  fi
  echo "${label}_CHECK_FAIL"
  return 1
}

if [ "$VENV_BOOTSTRAP" = "1" ]; then
  echo "BOOTSTRAP_REQUIREMENTS"
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -r requirements-public-benchmark.txt >/dev/null
fi

if ! run_import_check TENSORFLOW python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("tensorflow") else 1)
PY
then
  echo "INSTALL_TENSORFLOW"
  python -m pip install 'tensorflow>=2.16,<2.18' >/dev/null
fi

if ! run_import_check KERAS_MULTI_HEAD python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("keras_multi_head") else 1)
PY
then
  echo "INSTALL_KERAS_MULTI_HEAD"
  python -m pip install 'keras-multi-head==0.29.0' >/dev/null
fi

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

CMD=(
  "$VENV_DIR/bin/python"
  scripts/run_sota_crispr_hnn_public.py
  --repo-root .
  --datasets "$DATASETS"
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
