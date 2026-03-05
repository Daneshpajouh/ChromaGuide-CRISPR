#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

python3 -m venv .venv-public-benchmark
source .venv-public-benchmark/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-public-benchmark.txt

echo "Environment ready: $REPO_ROOT/.venv-public-benchmark"
