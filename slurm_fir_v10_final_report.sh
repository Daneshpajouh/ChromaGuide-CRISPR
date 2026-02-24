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

# Initialize module system (required on Alliance Canada compute nodes)
# Try multiple possible module paths on FIR
if [ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]; then
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
elif [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
else
    # Try to load modules directly (fallback for compute nodes)
    if command -v module >/dev/null 2>&1; then
        echo "Module command available"
    else
        echo "⚠️  Module system not available - using system Python"
    fi
fi

# Load required modules (Alliance Canada standard)
if command -v module >/dev/null 2>&1; then
    module load python/3.11.5 2>/dev/null || echo "⚠️  python/3.11.5 not available"
    module load cuda/12.2 2>/dev/null || echo "⚠️  cuda/12.2 not available"
else
    echo "⚠️  Module system not available - proceeding with system paths"
fi

export VENV_DIR=$SLURM_TMPDIR/env
python -m venv $VENV_DIR
source $VENV_DIR/bin/activate

pip install --find-links ~/.cache/pip-wheels torch transformers scipy scikit-learn pandas numpy h5py einops

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
