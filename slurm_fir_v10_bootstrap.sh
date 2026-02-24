#!/bin/bash
# SLURM job script for FIR - V10 Bootstrap Testing
# Alliance Canada standard: module load + virtualenv in SLURM_TMPDIR

#SBATCH --job-name=v10_bootstrap
#SBATCH --account=def-kwiese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --time=1:30:00
#SBATCH --output=logs/v10_bootstrap_%j.out
#SBATCH --error=logs/v10_bootstrap_%j.err
#SBATCH --mem=32G

set -euo pipefail

# Initialize module system (required on Alliance Canada compute nodes)
source /etc/profile.d/modules.sh

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
python -u scripts/run_bootstrap_testing.py \
    --multimodal_predictions results/multimodal_predictions.npz \
    --baseline_predictions results/chromeCRISPR_baseline.npz \
    --n_bootstrap 10000 \
    --output_path results/bootstrap_testing_results.json

echo "✓ Bootstrap testing complete"
