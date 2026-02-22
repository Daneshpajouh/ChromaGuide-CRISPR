#!/bin/bash
#SBATCH --job-name=bootstrap_testing
#SBATCH --account=def-kwiese_gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/bootstrap_testing_%j.log
#SBATCH --error=slurm_logs/bootstrap_testing_%j.err

module load cuda/12.2 python/3.11
source ~/env_chromaguide/bin/activate
export PYTHONPATH=~/chromaguide_experiments/src:$PYTHONPATH

cd ~/chromaguide_experiments

# Run bootstrap statistical testing
python3 scripts/run_bootstrap_testing.py \
    --baseline_predictions results/seq_only_baseline/predictions.csv \
    --multimodal_predictions results/multimodal/predictions.csv \
    --off_target_predictions results/off_target_v4/predictions.csv

echo "Bootstrap testing completed at $(date)"
