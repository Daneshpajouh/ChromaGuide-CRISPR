#!/bin/bash
# Test fix with BF16 and TF32 disabled
# Run with: bash test_fix.sh

echo "=== Testing H100 Fix (BF16 + NoTF32) ==="
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

# Clear checkpoints
rm -rf checkpoints/*.pth

echo "=== Running micro_4k test with fix (30s) ==="
timeout 120 python -m src.train --micro_4k 2>&1 | grep -E "(Epoch [0-9]+:|Loss|NaN|Device|MODE)" | head -30
