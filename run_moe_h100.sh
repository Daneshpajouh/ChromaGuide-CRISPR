#!/bin/bash
#SBATCH --job-name=crispro_moe
#SBATCH --account=def-kwiese
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/moe_training_%j.out
#SBATCH --error=logs/moe_training_%j.err

echo "=== CRISPRO-MAMBA-X MoE (Novelty) Training on H100 ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
echo ""

# Load Modules
echo "=== Loading Modules ==="
module load StdEnv/2023 python/3.10 scipy-stack cuda/12.2
echo "Modules loaded."

# Environment
export TRITON_CACHE_DIR=~/projects/def-kwiese/amird/tmp/triton_cache
export TORCH_CUDA_ARCH_LIST="9.0+PTX"
echo "TRITON_CACHE_DIR set to: $TRITON_CACHE_DIR"

# Activate Environment
echo "=== Activating mamba_h100 Environment ==="
source ~/projects/def-kwiese/amird/mamba_h100/bin/activate

# Navigate
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X || exit 1

# Run MoE Training (Novel Architecture)
# Utilizing --moe flag and reduced 50k subset for faster comparative validation
echo "=== Starting MoS Training (Comparison) ==="
python -m src.train --production --moe 2>&1

if [ $? -eq 0 ]; then
    echo "✅ MoE TRAINING SUCCESSFUL"
else
    echo "❌ MoE TRAINING FAILED (Code: $?)"
fi
