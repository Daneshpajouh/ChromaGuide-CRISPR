#!/bin/bash
#SBATCH --job-name=verify_real_h100
#SBATCH --account=def-kwiese
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/verify_real_h100_%j.out
#SBATCH --error=logs/verify_real_h100_%j.err

echo "=== CRISPRO-MAMBA-X Real Data Verification on H100 ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# 1. Load Modules
echo "=== Loading Modules ==="
module load StdEnv/2023 python/3.10 scipy-stack cuda/12.2
echo "Modules loaded."

# 2. Setup Environment Variables (Critical for H100)
export TRITON_CACHE_DIR=~/projects/def-kwiese/amird/tmp/triton_cache
mkdir -p $TRITON_CACHE_DIR
echo "TRITON_CACHE_DIR set to: $TRITON_CACHE_DIR"

export TORCH_CUDA_ARCH_LIST="9.0+PTX" # Good practice to keep
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# 3. Activate Verified Environment
echo "=== Activating mamba_h100 Environment ==="
source ~/projects/def-kwiese/amird/mamba_h100/bin/activate
which python
python --version

# 4. Run Training on Real Data (Medium 1k Subset)
echo "=== Starting Training (Real Data / Medium 1k) ==="
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X || exit 1
mkdir -p logs/training checkpoints

# Using --medium_1k flag as defined in src/train.py for safe initial testing
# This uses real data, not synthetic.
python -m src.train --medium_1k 2>&1

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✅ REAL DATA VERIFICATION SUCCESSFUL"
else
    echo "❌ REAL DATA VERIFICATION FAILED (Code: $exit_code)"
fi

exit $exit_code
