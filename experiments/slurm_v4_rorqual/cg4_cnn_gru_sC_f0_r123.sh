#!/bin/bash
#SBATCH --job-name=cg4_cnn__C_s123
#SBATCH --account=def-kwiese_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00

#SBATCH --output=~/scratch/chromaguide_v4/logs/cg4_cnn_gru_sC_f0_r123_%j.out
#SBATCH --error=~/scratch/chromaguide_v4/logs/cg4_cnn_gru_sC_f0_r123_%j.err

echo "============================================"
echo "ChromaGuide v4 Experiment"
echo "Job: cg4_cnn_gru_sC_f0_r123"
echo "Cluster: rorqual"
echo "Backbone: cnn_gru"
echo "Split: C (fold 0)"
echo "Seed: 123"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "============================================"

# Source environment
module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null
source ~/scratch/chromaguide_v2_env/bin/activate

# Ensure dependencies
pip install --no-index --upgrade pip 2>/dev/null
pip install omegaconf scipy 2>/dev/null

# Set environment variables
export TRANSFORMERS_CACHE=~/scratch/chromaguide_v4/model_cache
export HF_HOME=~/scratch/chromaguide_v4/model_cache
export TORCH_HOME=~/scratch/chromaguide_v4/model_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# Create directories
mkdir -p ~/scratch/chromaguide_v4/logs
mkdir -p ~/scratch/chromaguide_v4/results_v4
mkdir -p ~/scratch/chromaguide_v4/model_cache

cd ~/scratch/chromaguide_v4

# Run v4 experiment
echo "Starting v4 training..."
python experiments/train_experiment_v4.py \
    --backbone cnn_gru \
    --split C \
    --split-fold 0 \
    --seed 123 \
    --data-dir ~/scratch/chromaguide_v4/data \
    --output-dir ~/scratch/chromaguide_v4/results_v4 \
    --patience 30 \
    --gradient-clip 1.0 \
    --mixed-precision \
    --swa \
    --mixup-alpha 0.2 \
    --rc-augment \
    --label-smoothing 0.01 \
    --no-wandb \
    --model-cache-dir ~/scratch/chromaguide_v4/model_cache \
    --version v4

echo "============================================"
echo "Experiment complete: cg4_cnn_gru_sC_f0_r123"
echo "Date: $(date)"
echo "============================================"
