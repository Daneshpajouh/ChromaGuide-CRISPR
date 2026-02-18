#!/bin/bash
# Quick debug script for interactive GPU session
# Run with: bash debug_gpu.sh

echo "=== GPU Debug Session ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Load modules
module load StdEnv/2023 python/3.10 scipy-stack cuda/12.2

# Activate environment
source ~/projects/def-kwiese/amird/pytorch_env/bin/activate

# Change to project directory
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "=== Testing with --micro_4k (should work) ==="
timeout 60 python -m src.train --micro_4k 2>&1 | grep -E "(Epoch|Loss|NaN)" | head -20

echo ""
echo "=== Testing with --medium_1k (debugging NaN) ==="
timeout 120 python -m src.train --medium_1k 2>&1 | grep -E "(Epoch|Loss|NaN|DEBUG)" | head -30
