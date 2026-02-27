#!/bin/bash
#SBATCH --job-name=cg3_cadu_sC_f0_r42
#SBATCH --account=def-kwiese_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00

#SBATCH --output=~/scratch/chromaguide_v3/logs/cg3_cadu_sC_f0_r42_%j.out
#SBATCH --error=~/scratch/chromaguide_v3/logs/cg3_cadu_sC_f0_r42_%j.err

echo "============================================"
echo "ChromaGuide v3 Experiment"
echo "Job: cg3_cadu_sC_f0_r42"
echo "Cluster: narval"
echo "Backbone: caduceus"
echo "Split: C (fold 0)"
echo "Seed: 42"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# Source environment

module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null
source ~/scratch/chromaguide_v2_env/bin/activate

# Ensure dependencies
pip install --no-index --upgrade pip 2>/dev/null
pip install omegaconf scipy 2>/dev/null

# Set environment variables
export TRANSFORMERS_CACHE=~/scratch/chromaguide_v3/model_cache
export HF_HOME=~/scratch/chromaguide_v3/model_cache
export TORCH_HOME=~/scratch/chromaguide_v3/model_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# Create directories
mkdir -p ~/scratch/chromaguide_v3/logs
mkdir -p ~/scratch/chromaguide_v3/results_v3
mkdir -p ~/scratch/chromaguide_v3/model_cache

cd ~/scratch/chromaguide_v3

# Run experiment
echo "Starting training..."
python experiments/train_experiment_v3.py \
    --backbone caduceus \
    --split C \
    --split-fold 0 \
    --seed 42 \
    --data-dir ~/scratch/chromaguide_v3/data \
    --output-dir ~/scratch/chromaguide_v3/results_v3 \
    --loss-type logcosh \
    --optimizer adamax \
    --lambda-rank 0.1 \
    --patience 20 \
    --gradient-clip 1.0 \
    --mixed-precision \
    --no-wandb \
    --model-cache-dir ~/scratch/chromaguide_v3/model_cache \
    --version v3

echo "============================================"
echo "Experiment complete: cg3_cadu_sC_f0_r42"
echo "Date: $(date)"
echo "============================================"
