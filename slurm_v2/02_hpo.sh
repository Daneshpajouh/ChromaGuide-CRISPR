#!/bin/bash
#SBATCH --job-name=chromaguide-hpo
#SBATCH --account=def-kwiese
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/hpo_%j.out
#SBATCH --error=results/logs/hpo_%j.err

# ============================================================
# ChromaGuide: Hyperparameter Optimization (Optuna)
# SFU Fir Cluster - 1× A100-80GB
# Expected runtime: ~12-20 hours (50 trials)
# ============================================================

echo "============================================"
echo "ChromaGuide HPO"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "============================================"

module load python/3.11
module load cuda/12.2

source $HOME/chromaguide_env/bin/activate
cd $HOME/chromaguide

mkdir -p results/logs

# Run HPO with 50 Optuna trials
python -m chromaguide.cli hpo \
    --n-trials 50 \
    --split A \
    2>&1 | tee results/logs/hpo.log

echo "HPO complete: $(date)"
