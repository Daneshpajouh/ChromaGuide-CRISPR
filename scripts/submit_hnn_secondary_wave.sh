#!/bin/bash
set -euo pipefail
if [ $# -lt 3 ]; then
  echo "Usage: $0 <cluster> <canonical|transfer> <run_tag> [VAR=value ...]" >&2
  exit 1
fi
cluster="$1"
kind="$2"
run_tag="$3"
shift 3
case "$cluster" in
  fir)
    SSH_HOST=fir
    SBATCH=/opt/software/slurm/bin/sbatch
    REPO_DIR=/scratch/amird/chromaguide_experiments
    VENV_DIR=/scratch/amird/env_public_benchmark_hnn
    GPU_ARGS=(--gpus=h100:1)
    ACCOUNT=def-kwiese_gpu
    CUDA_MODULE=cuda/12.6
    ;;
  nibi)
    SSH_HOST=nibi
    SBATCH=/opt/software/slurm/24.11.7/bin/sbatch
    REPO_DIR=/scratch/amird/chromaguide_experiments
    VENV_DIR=/scratch/amird/env_public_benchmark_hnn
    GPU_ARGS=(--gpus=h100:1)
    ACCOUNT=def-kwiese_gpu
    CUDA_MODULE=cuda/12.6
    ;;
  narval)
    SSH_HOST=narval
    SBATCH=/opt/software/slurm/bin/sbatch
    REPO_DIR=/scratch/amird/chromaguide_experiments
    VENV_DIR=/scratch/amird/env_public_benchmark_hnn
    GPU_ARGS=(--gpus=a100:1)
    ACCOUNT=def-kwiese_gpu
    CUDA_MODULE=cuda/12.2
    ;;
  *)
    echo "Unsupported cluster: $cluster" >&2
    exit 1
    ;;
esac
if [ "$kind" = canonical ]; then
  REMOTE_SCRIPT=scripts/slurm_sota_crispr_hnn_public.sh
  defaults=(DATASETS=WT,ESP,HF FOLDS=5 EPOCHS=200 BATCH_SIZE=16 LR=0.0001 PATIENCE=10 MAX_ROWS=0 SEED=2024)
  OUTPUT_ROOT="results/public_benchmarks/${run_tag}"
  SUMMARY_JSON="${OUTPUT_ROOT}/SUMMARY.json"
elif [ "$kind" = transfer ]; then
  REMOTE_SCRIPT=scripts/slurm_sota_crispr_hnn_transfer_public.sh
  defaults=(SOURCE_DATASET=WT TARGET_DATASET=HL60 FOLDS=5 EPOCHS=200 BATCH_SIZE=16 LR=0.0001 PATIENCE=10 MAX_ROWS=0 SEED=2024)
  OUTPUT_ROOT="results/public_benchmarks/${run_tag}"
  SUMMARY_JSON="${OUTPUT_ROOT}/SUMMARY.json"
else
  echo "Unsupported kind: $kind" >&2
  exit 1
fi
ENVVARS=(REPO_DIR="$REPO_DIR" VENV_DIR="$VENV_DIR" CUDA_MODULE="$CUDA_MODULE" RUN_TAG="$run_tag" OUTPUT_ROOT="$OUTPUT_ROOT" SUMMARY_JSON="$SUMMARY_JSON")
for kv in "${defaults[@]}"; do ENVVARS+=("$kv"); done
for kv in "$@"; do ENVVARS+=("$kv"); done
remote_env=""
for kv in "${ENVVARS[@]}"; do remote_env+="$kv "; done
remote_cmd="cd $REPO_DIR && ${remote_env}${SBATCH} --parsable --account=${ACCOUNT} --time=24:00:00 ${GPU_ARGS[*]} --job-name=${run_tag} --output=slurm_logs/${run_tag}_%j.out --error=slurm_logs/${run_tag}_%j.err ${REMOTE_SCRIPT}"
ssh "$SSH_HOST" "$remote_cmd"
