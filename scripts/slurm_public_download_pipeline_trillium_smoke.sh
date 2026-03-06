#!/bin/bash
#SBATCH --job-name=pub_downloads_smoke
#SBATCH --account=def-kwiese
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/amird/chromaguide_experiments/slurm_logs/public_downloads_smoke_%j.out
#SBATCH --error=/scratch/amird/chromaguide_experiments/slurm_logs/public_downloads_smoke_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/${USER}/chromaguide_experiments}"
export REPO_DIR
export MODE="smoke"
export SMOKE_FAST="1"
export SMOKE_SKIP_CCLMOFF_HEAD="1"

mkdir -p "$REPO_DIR/slurm_logs"
cd "$REPO_DIR"

exec bash "$REPO_DIR/scripts/slurm_public_download_pipeline.sh"
