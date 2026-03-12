#!/bin/bash
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "Usage: $0 <cluster>" >&2
  exit 1
fi
cluster="$1"
case "$cluster" in
  fir)
    SSH_HOST=fir
    SBATCH=/opt/software/slurm/bin/sbatch
    REPO_DIR=/scratch/amird/chromaguide_experiments
    VENV_DIR=/scratch/amird/env_public_benchmark_hnn
    EXTRA_GPU='--gpus=h100:1'
    CUDA_MODULE='cuda/12.6'
    ;;
  nibi)
    SSH_HOST=nibi
    SBATCH=/opt/software/slurm/24.11.7/bin/sbatch
    REPO_DIR=/scratch/amird/chromaguide_experiments
    VENV_DIR=/scratch/amird/env_public_benchmark_hnn
    EXTRA_GPU='--gpus=h100:1'
    CUDA_MODULE='cuda/12.6'
    ;;
  narval)
    SSH_HOST=narval
    SBATCH=/opt/software/slurm/bin/sbatch
    REPO_DIR=/scratch/amird/chromaguide_experiments
    VENV_DIR=/scratch/amird/env_public_benchmark_hnn
    EXTRA_GPU='--gpus=a100:1'
    CUDA_MODULE='cuda/12.2'
    ;;
  *)
    echo "Unsupported cluster: $cluster" >&2
    exit 1
    ;;
esac
RUN_TAG="smoke_hnn_gpucheck_${cluster}_$(date +%Y%m%d_%H%M%S)"
REMOTE_CMD="cd ${REPO_DIR} && REPO_DIR=${REPO_DIR} VENV_DIR=${VENV_DIR} CUDA_MODULE=${CUDA_MODULE} DATASETS=WT FOLDS=1 EPOCHS=1 BATCH_SIZE=16 LR=0.0001 PATIENCE=2 MAX_ROWS=200 SEED=2024 RUN_TAG=${RUN_TAG} OUTPUT_ROOT=results/public_benchmarks/${RUN_TAG} SUMMARY_JSON=results/public_benchmarks/${RUN_TAG}/SUMMARY.json ${SBATCH} --parsable --account=def-kwiese_gpu --time=00:10:00 ${EXTRA_GPU} --job-name=hnn_smoke_${cluster} --output=slurm_logs/hnn_smoke_%j.out --error=slurm_logs/hnn_smoke_%j.err scripts/slurm_sota_crispr_hnn_public.sh"
ssh "$SSH_HOST" "$REMOTE_CMD"
