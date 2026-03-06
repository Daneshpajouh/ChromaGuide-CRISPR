#!/usr/bin/env bash
set -euo pipefail

# Create isolated conda envs for upstream SOTA baseline reproduction.
# Usage:
#   scripts/bootstrap_sota_external_envs.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[bootstrap] root: ${ROOT_DIR}"
echo "[bootstrap] conda: $(command -v conda)"
echo "[bootstrap] mamba: $(command -v mamba)"

create_env_if_missing() {
  local env_name="$1"
  local pyver="$2"
  if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    echo "[bootstrap] env exists: ${env_name}"
  else
    echo "[bootstrap] creating env: ${env_name} (python=${pyver})"
    mamba create -y -n "${env_name}" "python=${pyver}" pip
  fi
}

env_python() {
  local env_name="$1"
  echo "/Users/studio/mambaforge/envs/${env_name}/bin/python"
}

install_into_env() {
  local env_name="$1"
  shift
  local py_bin
  py_bin="$(env_python "${env_name}")"
  if [[ ! -x "${py_bin}" ]]; then
    echo "[bootstrap] missing python binary for ${env_name}: ${py_bin}" >&2
    exit 1
  fi
  env -u PYTHONPATH PYTHONNOUSERSITE=1 "${py_bin}" -m pip install --upgrade pip
  env -u PYTHONPATH PYTHONNOUSERSITE=1 "${py_bin}" -m pip install "$@"
}

# CRISPR_HNN (TensorFlow-based upstream)
create_env_if_missing "sota_crispr_hnn" "3.10"
echo "[bootstrap] installing CRISPR_HNN deps"
install_into_env "sota_crispr_hnn" \
  numpy pandas scipy scikit-learn keras-multi-head \
  tensorflow-macos==2.15.0

# crispAI (uncertainty baseline)
create_env_if_missing "sota_crispai" "3.10"
echo "[bootstrap] installing crispAI deps"
install_into_env "sota_crispai" \
  torch numpy pandas scipy scikit-learn biopython genomepy loguru tqdm

echo "[bootstrap] done"
