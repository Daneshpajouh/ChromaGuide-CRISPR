#!/bin/bash
#SBATCH --job-name=chromaguide_seq_baseline
#SBATCH --account=def-kwiese
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/baseline_unbuffered_%j.log
#SBATCH --error=slurm_logs/baseline_unbuffered_%j.err

module load cuda/12.2 python/3.11
source ~/env_chromaguide/bin/activate

# Force unbuffered output and offline mode
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=~/chromaguide_experiments/src:$PYTHONPATH

cd ~/chromaguide_experiments
mkdir -p slurm_logs results/seq_only_baseline

# Run with unbuffered output (-u flag)
python -u scripts/train_on_real_data_v2.py \
    --backbone cnn_gru \
    --epochs 30 \
    --device cuda
