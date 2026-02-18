#!/bin/bash
#SBATCH --job-name=benchmark_sota
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# SOTA Benchmark Validation (Cluster Execution)
# This job runs the full benchmark suite on DeepMEns and RNAGenesis
# to verify SOTA targets on the complete test set.

echo "SOTA Benchmark Suite - Cluster Execution"
echo "Started: $(date)"

cd /path/to/Proposal  # UPDATE THIS PATH

# Activate environment
source activate crispr  # Or your env name

# Run Benchmark
python3 src/benchmark_suite.py

echo "Completed: $(date)"
