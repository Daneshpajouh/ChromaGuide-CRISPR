#!/bin/bash
#SBATCH --job-name=cg-cnn_gr-B-s456
#SBATCH --account=def-kwiese_gpu
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=results/cnn_gru_splitB_seed456/slurm-%j.out
#SBATCH --error=results/cnn_gru_splitB_seed456/slurm-%j.err

# ============================================================
# ChromaGuide: cnn_gru | Split B | Seed 456
# Cluster: nibi (GPU)
# ============================================================

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Backbone: cnn_gru"
echo "Split: B"
echo "Seed: 456"
echo "Cluster: nibi"
echo "==================="

# Load modules
module load StdEnv/2023 python/3.11.5 scipy-stack cuda/12.2

# Activate virtual environment
source ~/scratch/chromaguide_v2_env/bin/activate

# Environment variables
export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=~/scratch/.cache/huggingface
export HF_HOME=~/scratch/.cache/huggingface
export TORCH_HOME=~/scratch/.cache/torch
export WANDB_MODE=offline
export OMP_NUM_THREADS=4

# Navigate to project
cd ~/scratch/chromaguide_v2

# Create output directory
mkdir -p results/cnn_gru_splitB_seed456

echo ""
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
python -c 'import torch; print(f"GPU: {torch.cuda.get_device_name(0)}") if torch.cuda.is_available() else print("No GPU")'
echo ""

# Run training
python experiments/train_experiment.py \
    --backbone cnn_gru \
    --split B \
    --split-fold 0 \
    --seed 456 \
    --data-dir ~/scratch/chromaguide_v2/data \
    --output-dir ~/scratch/chromaguide_v2/results \
    --no-wandb \
    --patience 15 \
    --gradient-clip 1.0 \
    --epochs 100 --batch-size 256

EXIT_CODE=$?

echo ""
echo "===== JOB COMPLETE ====="
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

if [ -f results/cnn_gru_splitB_seed456/results.json ]; then
    echo ""
    echo "=== RESULTS ==="
    python -c "
import json
with open('results/cnn_gru_splitB_seed456/results.json') as f:
    r = json.load(f)
print(f'Spearman: {r["test_metrics"]["spearman"]:.4f}')
print(f'Pearson:  {r["test_metrics"]["pearson"]:.4f}')
print(f'ECE:      {r["test_metrics"]["ece"]:.4f}')
print(f'Coverage: {r["conformal"]["coverage"]:.4f}')
print(f'Time:     {r["training_time_seconds"]/60:.1f} min')
"
fi

exit $EXIT_CODE
