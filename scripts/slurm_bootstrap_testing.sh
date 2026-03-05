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
    --results_a results/gold_results_on_target.csv \
    --results_b results/sequence_baseline_results.csv

echo "Bootstrap testing completed at $(date)"
