#!/bin/bash
#SBATCH --job-name=generate_final_report
#SBATCH --account=def-kwiese_gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/generate_final_report_%j.log
#SBATCH --error=slurm_logs/generate_final_report_%j.err

module load python/3.11
source ~/env_chromaguide/bin/activate
export PYTHONPATH=~/chromaguide_experiments/src:$PYTHONPATH

cd ~/chromaguide_experiments

# Generate comprehensive final report with all results
python3 scripts/generate_final_report.py \
    --results_dir results/ \
    --output_file results/FINAL_REPORT.md

echo "Final report generation completed at $(date)"
echo "Report saved to: results/FINAL_REPORT.md"
