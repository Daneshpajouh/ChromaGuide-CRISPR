#!/bin/bash
#SBATCH --job-name=chromaguide-multimodal-cnn_gru
#SBATCH --account=def-kwiese_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/multimodal_cnn_gru_%j.out
#SBATCH --error=slurm_logs/multimodal_cnn_gru_%j.err

module load cuda/12.2 python/3.11
source ~/env_chromaguide/bin/activate

# Force OFFLINE mode - prevent all network access
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

# Set path to cached models
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/hub"

export PYTHONPATH=~/chromaguide_experiments/src:$PYTHONPATH

cd ~/chromaguide_experiments
mkdir -p slurm_logs

# Run with unbuffered output using CNN-GRU backbone (stable, no DNABERT-2 meta device issues)
python -u scripts/train_on_real_data_v2.py \
    --backbone cnn_gru \
    --epochs 30 \
    --device cuda
