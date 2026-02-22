#!/bin/bash
#SBATCH --job-name=calibrate_conformal
#SBATCH --account=def-kwiese_gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/calibrate_conformal_%j.log
#SBATCH --error=slurm_logs/calibrate_conformal_%j.err

module load cuda/12.2 python/3.11
source ~/env_chromaguide/bin/activate
export PYTHONPATH=~/chromaguide_experiments/src:$PYTHONPATH

cd ~/chromaguide_experiments

# Run conformal calibration with trained models
python3 scripts/calibrate_conformal.py \
    --model_path best_model_on_target.pt \
    --data_path data/real/merged.csv \
    --backbone dnabert2 \
    --alpha 0.1

echo "Conformal calibration completed at $(date)"
