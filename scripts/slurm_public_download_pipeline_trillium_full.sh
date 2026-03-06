#!/bin/bash
#SBATCH --job-name=pub_downloads_full
#SBATCH --account=def-kwiese
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/amird/chromaguide_experiments/slurm_logs/public_downloads_full_%j.out
#SBATCH --error=/scratch/amird/chromaguide_experiments/slurm_logs/public_downloads_full_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/${USER}/chromaguide_experiments}"
export REPO_DIR
export MODE="full"
export SMOKE_SKIP_CCLMOFF_HEAD="1"

mkdir -p "$REPO_DIR/slurm_logs"
cd "$REPO_DIR"

exec bash "$REPO_DIR/scripts/slurm_public_download_pipeline.sh"
