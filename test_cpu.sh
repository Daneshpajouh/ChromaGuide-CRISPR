#!/bin/bash
# Test CPU-only training to confirm environment stability
# Run with: bash test_cpu.sh

echo "=== Testing CPU Only (Masking CUDA) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# Mask CUDA to force CPU
export CUDA_VISIBLE_DEVICES=""

# Load modules
module load StdEnv/2023 python/3.10 scipy-stack cuda/12.2

# Activate environment
source ~/projects/def-kwiese/amird/pytorch_env/bin/activate

# Change to project directory
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Clear checkpoints
rm -rf checkpoints/*.pth

echo "=== Running micro_4k test on CPU (30s) ==="
# Should be slow but stable (no NaN)
timeout 60 python -m src.train --micro_4k 2>&1 | grep -E "(Epoch [0-9]+:|Loss|NaN|Device|MODE)" | head -30
