#!/bin/bash
# SLURM job script for Fir (Alliance Canada) — DNABERT -> Mamba training
# Login host: fir.alliancecan.ca
# Request 1 node with 4x H100 SXM5 GPUs and 48 CPU cores

#SBATCH --job-name=dnabert_mamba
#SBATCH --account=def-amird
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus=h100:4
#SBATCH --gres=gpu:h100:4
#SBATCH --time=168:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amird@your.email.example

# Optional: adjust memory if your cluster requires it
#SBATCH --mem=480G

set -euo pipefail

echo "Job $SLURM_JOB_ID starting on $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# Load recommended modules (names may vary on Fir — adjust if needed)
module purge
module load gcc/12
module load cuda/12.1
module load nccl
module load python/3.11

# Print module and GPU info for debugging
module list || true
nvidia-smi || true

# Use a virtualenv in the job's local directory (or change to a persistent env)
VENV_DIR="$HOME/venvs/dnabert_mamba_${SLURM_JOB_ID}"
mkdir -p "$HOME/venvs"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel
# Install minimal requirements — modify to point to your project's requirements file
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
fi
# Ensure critical packages for DNABERT are present
pip install --no-cache-dir einops transformers accelerate

# Create HF cache on fast parallel storage (SCRATCH) if available
if [ -n "${SCRATCH:-}" ]; then
  export HF_HOME="$SCRATCH/hf_cache"
  export TRANSFORMERS_CACHE="$SCRATCH/hf_cache/transformers"
  mkdir -p "$TRANSFORMERS_CACHE"
  echo "Using HF cache: $TRANSFORMERS_CACHE"
fi

# Recommended environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
export HF_TOKEN=${HF_TOKEN:-}

# Move to repository root (assumes script is submitted from repo root)
cd "$SLURM_SUBMIT_DIR"

# Create logs directory
mkdir -p logs

# Run the training with srun to ensure proper tracking
SRUN_CMD=(srun --mpi=pmix_v3 --cpu-bind=cores)

TRAIN_CMD=(/Users/studio/mambaforge/envs/chromaguide/bin/python -u src/train_dnabert_mamba.py \
  --dnabert_name zhihan1996/DNABERT-2-117M \
  --trust_remote_code \
  --epochs 30 \
  --batch_size 16 \
  --scheduler plateau)

echo "Running: ${SRUN_CMD[*]} ${TRAIN_CMD[*]}"

"${SRUN_CMD[@]}