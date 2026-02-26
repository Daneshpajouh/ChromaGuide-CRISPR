#!/bin/bash
#SBATCH --job-name=chromaguide-offtarget
#SBATCH --account=def-kwiese
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=results/logs/offtarget_%j.out
#SBATCH --error=results/logs/offtarget_%j.err

# ============================================================
# ChromaGuide: Off-Target Module Training
# ============================================================

echo "============================================"
echo "ChromaGuide Off-Target Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "============================================"

module load python/3.11
module load cuda/12.2

source $HOME/chromaguide_env/bin/activate
cd $HOME/chromaguide

mkdir -p results/logs

python -m chromaguide.cli offtarget 2>&1 | tee results/logs/offtarget.log

echo "Off-target training complete: $(date)"
