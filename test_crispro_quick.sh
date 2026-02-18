#!/bin/bash
#SBATCH --job-name=test_crispro
#SBATCH --account=def-kwiese
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_crispro_%j.out
#SBATCH --error=logs/test_crispro_%j.err

echo "=== CRISPRO-MAMBA-X Micro Test (--micro_4k) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Load modules
echo "=== Loading Modules ==="
module load python/3.10
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

echo "=== Running Micro-Scale Test (20 samples, 10 epochs) ==="
echo "This tests train.py with --micro_4k flag"
echo ""

# Run training with micro_4k mode
python -m src.train --micro_4k 2>&1 || {
    echo ""
    echo "=== Training Failed ==="
    exit 1
}

echo ""
echo "=== Test Complete ==="
echo "End time: $(date)"
echo ""
echo "✅ Micro test succeeded! Ready for full training."
