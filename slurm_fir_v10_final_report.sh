#!/bin/bash
# SLURM job script for FIR - V10 Final Report Generation
# Purpose: Aggregate all results and generate comprehensive proposal report

#SBATCH --job-name=v10_report
#SBATCH --account=def-kwiese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/v10_final_report_%j.out
#SBATCH --error=logs/v10_final_report_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=16G

set -euo pipefail

echo "=== Final Report Generation ==="
cd ~/chromaguide

python3 -u scripts/generate_final_report.py \
    --off_target_results results/off_target_results.json \
    --multimodal_results results/multimodal_results.json \
    --conformal_results results/conformal_calibration.json \
    --bootstrap_results results/bootstrap_testing_results.json \
    --designer_results results/designer_scores_results.json \
    --ablation_fusion_results results/ablation_fusion_results.json \
    --ablation_modality_results results/ablation_modality_results.json \
    --output_path results/final_proposal_report.json \
    --html_output results/final_proposal_report.html

echo "✓ Final report complete"
