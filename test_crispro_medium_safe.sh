#!/bin/bash
#SBATCH --job-name=test_medium_safe
#SBATCH --account=def-kwiese
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/test_medium_safe_%j.out
#SBATCH --error=logs/test_medium_safe_%j.err

echo "=== CRISPRO-MAMBA-X Medium Test (--medium_1k mode) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""
echo "Configuration:"
echo "  - 999 samples (full mini dataset)"
echo "  - Learning Rate: 1e-4 (10x safer than production)"
echo "  - Gradient Clipping: 0.5 (aggressive)"
echo "  - Weight Decay: 0.01"
echo "  - Epochs: 30"
echo ""

# Load modules
echo "=== Loading Modules ==="
module load StdEnv/2023 python/3.10 scipy-stack cuda/12.2
echo ""

# Activate environment
echo "=== Activating Virtual Environment ==="
source ~/projects/def-kwiese/amird/pytorch_env/bin/activate
python --version
echo ""

# Change to project directory and set PYTHONPATH
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X || exit 1
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "=== Directory: $(pwd) ==="
echo ""

# Create directories
mkdir -p checkpoints logs/training data/processed

echo "=== Running Medium-Scale Test with Safe Hyperparameters ==="
echo "This uses --medium_1k flag for safer training"
echo ""

# Run training with medium_1k mode (safer hyperparameters)
python -m src.train --medium_1k 2>&1 || {
    echo ""
    echo "=== Training Failed ==="
    exit 1
}

echo ""
echo "=== Test Complete ==="
echo "End time: $(date)"
echo ""
echo "✅ Medium test with safe hyperparameters succeeded!"
