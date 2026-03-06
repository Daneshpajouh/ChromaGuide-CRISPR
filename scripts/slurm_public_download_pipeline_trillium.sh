#!/bin/bash
#SBATCH --job-name=pub_downloads
#SBATCH --account=def-kwiese
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/amird/chromaguide_experiments/slurm_logs/public_downloads_%j.out
#SBATCH --error=/scratch/amird/chromaguide_experiments/slurm_logs/public_downloads_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/${USER}/chromaguide_experiments}"
export REPO_DIR

mkdir -p "$REPO_DIR/slurm_logs"
cd "$REPO_DIR"

exec bash "$REPO_DIR/scripts/slurm_public_download_pipeline.sh"
