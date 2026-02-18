#!/bin/bash
#SBATCH --job-name=test_1k
#SBATCH --account=def-kwiese
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/test_medium_%j.out
#SBATCH --error=logs/test_medium_%j.err

echo "=== CRISPRO-MAMBA-X Medium Test (999 samples) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
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

echo "=== Running Medium-Scale Test (999 samples, 50 epochs) ==="
echo "This runs without --micro_4k flag to use full mini dataset"
echo ""

# Run training without micro_4k mode (uses 999 samples from mini dataset)
python -m src.train 2>&1 || {
    echo ""
    echo "=== Training Failed ==="
    exit 1
}

echo ""
echo "=== Test Complete ==="
echo "End time: $(date)"
echo ""
echo "✅ Medium test succeeded! Ready for full production training."
