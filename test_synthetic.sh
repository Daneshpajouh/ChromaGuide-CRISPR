#!/bin/bash
# Test Synthetic Data on GPU
# Run with: bash test_synthetic.sh

echo "=== Testing Synthetic Data (H100) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Load modules
module load StdEnv/2023 python/3.10 scipy-stack cuda/12.2

# Activate environment
source ~/projects/def-kwiese/amird/pytorch_env/bin/activate

# Change to project directory
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Clear checkpoints
rm -rf checkpoints/*.pth

echo "=== Running synthetic test (60s) ==="
# Using unbuffered output to see progress immediately
timeout 60 python -u -m src.train --synthetic 2>&1 | head -50
