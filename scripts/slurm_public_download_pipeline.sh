#!/bin/bash
#SBATCH --job-name=pub_downloads
#SBATCH --account=def-kwiese_gpu
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/public_downloads_%j.out
#SBATCH --error=slurm_logs/public_downloads_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/chromaguide_experiments}"
VENV_DIR="${VENV_DIR:-$HOME/env_public_benchmark}"
USE_VENV="${USE_VENV:-0}"
MODE="${MODE:-smoke}" # smoke | full
SMOKE_FAST="${SMOKE_FAST:-0}"
JOB_ID="${SLURM_JOB_ID:-local}"
RUN_TAG="${RUN_TAG:-public_downloads_${MODE}_${JOB_ID}}"
VENV_BOOTSTRAP="${VENV_BOOTSTRAP:-0}"
RESULT_DIR="${RESULT_DIR:-results/public_benchmarks/download_pipeline/${RUN_TAG}}"
SOTA_STATUS_JSON="${SOTA_STATUS_JSON:-${RESULT_DIR}/sota_source_acquisition_status.json}"
PUBLIC_ACQ_JSON="${PUBLIC_ACQ_JSON:-${RESULT_DIR}/public_source_fetch_status.json}"
STAGING_JSON="${STAGING_JSON:-${RESULT_DIR}/local_dataset_map.json}"
CCLMOFF_SMOKE_BYTES="${CCLMOFF_SMOKE_BYTES:-1048576}"
CCLMOFF_URL="${CCLMOFF_URL:-https://ndownloader.figshare.com/files/49344577}"
CCLMOFF_CURL_MAX_TIME="${CCLMOFF_CURL_MAX_TIME:-120}"
SMOKE_SKIP_CCLMOFF_HEAD="${SMOKE_SKIP_CCLMOFF_HEAD:-0}"
SOTA_SKIP_URL_CHECKS="${SOTA_SKIP_URL_CHECKS:-1}"

mkdir -p "$REPO_DIR/slurm_logs"
mkdir -p "$REPO_DIR/${RESULT_DIR}"
cd "$REPO_DIR"

if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load python/3.11 || true
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ "$USE_VENV" = "1" ]; then
  if [ ! -f "$VENV_DIR/bin/activate" ]; then
    if [ -d "$VENV_DIR" ]; then
      python3 -m venv --clear "$VENV_DIR"
    else
      python3 -m venv "$VENV_DIR"
    fi
  fi
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN="python"
  if [ "$VENV_BOOTSTRAP" = "1" ]; then
    python -m pip install --upgrade pip >/dev/null
    python -m pip install -r requirements-public-benchmark.txt >/dev/null
  fi
fi

unset PYTHONPATH || true
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

echo "[$(date -Is)] mode=${MODE} repo=${REPO_DIR} result_dir=${RESULT_DIR}"
echo "Python: $($PYTHON_BIN --version)"

if [ "$MODE" = "smoke" ]; then
  if [ "$SMOKE_FAST" = "1" ]; then
    "$PYTHON_BIN" scripts/acquire_public_benchmarks.py --help >/dev/null
    "$PYTHON_BIN" scripts/acquire_sota_model_sources.py --help >/dev/null
    mkdir -p "$(dirname "$SOTA_STATUS_JSON")"
    cat > "${RESULT_DIR}/smoke_fast_status.json" <<EOF
{"mode":"smoke_fast","job_id":"${JOB_ID}","repo_dir":"${REPO_DIR}"}
EOF
    if [ "$SMOKE_SKIP_CCLMOFF_HEAD" != "1" ]; then
      curl -fL --connect-timeout 20 --max-time "$CCLMOFF_CURL_MAX_TIME" \
        --range "0-$((CCLMOFF_SMOKE_BYTES-1))" "$CCLMOFF_URL" \
        -o "${RESULT_DIR}/cclmoff_head_${CCLMOFF_SMOKE_BYTES}.bin"
    fi
    echo "[$(date -Is)] smoke-fast completed"
    exit 0
  fi

  "$PYTHON_BIN" scripts/acquire_public_benchmarks.py --repo-root . --output-json "$PUBLIC_ACQ_JSON"
  "$PYTHON_BIN" scripts/prepare_public_benchmark_inputs.py --repo-root . --output-json "$STAGING_JSON"
  "$PYTHON_BIN" scripts/acquire_sota_model_sources.py \
    --repo-root . \
    --output "$SOTA_STATUS_JSON" \
    --dated-output "$SOTA_STATUS_JSON" \
    --skip-url-checks
  mkdir -p "$(dirname "$SOTA_STATUS_JSON")"
  if [ "$SMOKE_SKIP_CCLMOFF_HEAD" != "1" ]; then
    curl -fL --connect-timeout 20 --max-time "$CCLMOFF_CURL_MAX_TIME" \
      --range "0-$((CCLMOFF_SMOKE_BYTES-1))" "$CCLMOFF_URL" \
      -o "${RESULT_DIR}/cclmoff_head_${CCLMOFF_SMOKE_BYTES}.bin"
  fi
  echo "[$(date -Is)] smoke completed"
  exit 0
fi

if [ "$MODE" != "full" ]; then
  echo "Unsupported MODE=${MODE}. Expected smoke or full." >&2
  exit 1
fi

"$PYTHON_BIN" scripts/acquire_public_benchmarks.py --repo-root . --output-json "$PUBLIC_ACQ_JSON"
"$PYTHON_BIN" scripts/prepare_public_benchmark_inputs.py --repo-root . --output-json "$STAGING_JSON"

SOTA_ARGS=(
  --repo-root .
  --output "$SOTA_STATUS_JSON"
  --dated-output "$SOTA_STATUS_JSON"
  --update-existing
)
if [ "$SOTA_SKIP_URL_CHECKS" = "1" ]; then
  SOTA_ARGS+=(--skip-url-checks)
fi
"$PYTHON_BIN" scripts/acquire_sota_model_sources.py "${SOTA_ARGS[@]}"

bash scripts/download_cclmoff_primary.sh
"$PYTHON_BIN" scripts/build_public_off_target_frames.py

echo "[$(date -Is)] full download pipeline completed"
