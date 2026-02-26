#!/bin/bash
#SBATCH --job-name=chromaguide-data
#SBATCH --account=def-kwiese
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/data_%j.out
#SBATCH --error=results/logs/data_%j.err

# ============================================================
# ChromaGuide: Data Acquisition & Preprocessing
# SFU Fir Cluster
# ============================================================

echo "============================================"
echo "ChromaGuide Data Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================"

# Load modules
module load python/3.11
module load git/2.40

# Activate virtual environment
source $HOME/chromaguide_env/bin/activate

# Navigate to project
cd $HOME/chromaguide

# Create directories
mkdir -p results/logs data/raw data/processed

# Step 1: Download all datasets
echo "--- Step 1: Data Acquisition ---"
python -m chromaguide.cli data --stage download 2>&1 | tee results/logs/download.log

# Step 2: Preprocess
echo "--- Step 2: Preprocessing ---"
python -m chromaguide.cli data --stage preprocess 2>&1 | tee results/logs/preprocess.log

# Step 3: Build splits
echo "--- Step 3: Split Construction ---"
python -m chromaguide.cli data --stage splits 2>&1 | tee results/logs/splits.log

echo "============================================"
echo "Data pipeline complete: $(date)"
echo "============================================"
