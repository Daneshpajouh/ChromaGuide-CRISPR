#!/bin/bash
#SBATCH --job-name=chromaguide-5x2cv
#SBATCH --account=def-kwiese
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=results/logs/5x2cv_%j.out
#SBATCH --error=results/logs/5x2cv_%j.err

# ============================================================
# ChromaGuide: 5×2 Cross-Validation for Statistical Testing
# This runs 10 full training cycles (5 folds × 2 repeats)
# Expected runtime: ~24-48 hours on 1× A100
# ============================================================

echo "============================================"
echo "ChromaGuide 5×2 Cross-Validation"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "============================================"

module load python/3.11
module load cuda/12.2

source $HOME/chromaguide_env/bin/activate
cd $HOME/chromaguide

mkdir -p results/logs results/cv_results

python scripts/run_5x2cv.py 2>&1 | tee results/logs/5x2cv.log

echo "5×2cv complete: $(date)"
