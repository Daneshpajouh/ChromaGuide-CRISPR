#!/bin/bash
# SLURM job script for FIR - V10 Final Report Generation
# Alliance Canada standard: module load + virtualenv in SLURM_TMPDIR

#SBATCH --job-name=v10_report
#SBATCH --account=def-kwiese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/v10_final_report_%j.out
#SBATCH --error=logs/v10_final_report_%j.err
#SBATCH --mem=16G

set -euo pipefail

module load python/3.11.5
module load cuda/12.2

export VENV_DIR=$SLURM_TMPDIR/env
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

pip install torch torchvision --no-index
pip install transformers scipy scikit-learn pandas numpy h5py --no-index

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCH_HOME=~/.cache/torch

cd ~/chromaguide
python -u scripts/generate_final_report.py \
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
