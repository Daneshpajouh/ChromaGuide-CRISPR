#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

python3 scripts/acquire_public_benchmarks.py --repo-root .
python3 scripts/prepare_public_benchmark_inputs.py --repo-root .
python3 scripts/evaluate_public_on_target_readiness.py --repo-root .
python3 scripts/evaluate_public_benchmark_readiness.py --repo-root .
python3 scripts/run_public_benchmark_harness.py --repo-root .

cat <<'MSG'

Acquisition and staging complete.

Manual follow-ups still required for the modern primary off-target suite:
1. Download the CCLMoff Figshare compiled bundle and stage it under data/public_benchmarks/off_target/primary_cclmoff.
2. Extract DNABERT-Epi supplement tables for the unified classifier benchmark.
3. Extract or reconstruct CHANGE-seq processed site tables from GEO/supplement for the regression and uncertainty frame.

Training/evaluation of the actual public benchmarks still requires a Python env with numpy, pandas, and torch.
MSG
