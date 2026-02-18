#!/bin/bash
#SBATCH --job-name=crispro_train
#SBATCH --account=def-kwiese
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/crispro_train_%j.out
#SBATCH --error=logs/crispro_train_%j.err

echo "=== CRISPRO-MAMBA-X Training Job ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Working Directory: $(pwd)"
echo ""

# Load modules
echo "=== Loading Modules ==="
module load python/3.10
module list
echo ""

# Activate PyTorch environment
echo "=== Activating Virtual Environment ==="
source ~/projects/def-kwiese/amird/pytorch_env/bin/activate
which python
python --version
echo ""

# Verify GPU availability
echo "=== Verifying GPU ==="
python << EOF
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
EOF
echo ""

# Change to project directory
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X || exit 1

# Create necessary directories
mkdir -p data/processed
mkdir -p checkpoints
mkdir -p logs/training

echo "=== Training Configuration ==="
echo "Model: CRISPRO-MAMBA-X"
echo "Epochs: 50"
echo "Batch Size: 256"
echo "Learning Rate: 1e-4"
echo ""

# Run training
echo "=== Starting Training ==="
python -m src.train \
    --epochs 50 \
    --batch_size 256 \
    --learning_rate 1e-4 \
    --model_type mamba2 \
    --checkpoint_dir checkpoints/ \
    --log_dir logs/training/ \
    --num_workers 4 \
    --save_every 5 \
    --device cuda

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
echo "Check logs at: logs/training/"
echo "Check checkpoints at: checkpoints/"
