# Cluster Job Submission Instructions

## Prerequisites
You need access to a SLURM cluster (e.g., university HPC, AWS ParallelCluster, etc.)

## Step 1: Upload Code to Cluster
```bash
# From your Mac, sync the code to the cluster
rsync -avz --exclude='*.pt' --exclude='data/' \
  /Users/studio/Desktop/PhD/Proposal/ \
  username@cluster.university.edu:/scratch/username/Proposal/
```

## Step 2: Update SLURM Script Paths
Edit `run_benchmark_cluster.slurm` and update:
- Line 17: Change `/path/to/Proposal` to your cluster path (e.g., `/scratch/username/Proposal`)
- Line 20: Change environment name if needed (e.g., `conda activate crispr`)

## Step 3: Submit Job
```bash
# SSH to cluster
ssh username@cluster.university.edu

# Navigate to project
cd /scratch/username/Proposal

# Create logs directory
mkdir -p logs

# Submit job
sbatch run_benchmark_cluster.slurm

# Check job status
squeue -u username

# Monitor output (once job starts)
tail -f logs/benchmark_*.out
```

## Expected Output
The benchmark suite will generate:
1. `benchmark_results.json` with final metrics
2. Console output with Spearman Rho and Success Rate

## Alternative: Run Manually on Cluster
If you don't have SLURM, you can run directly:
```bash
# SSH to cluster with GPU
ssh username@gpu-node.cluster.edu

# Activate environment
conda activate crispr

# Run benchmark
cd /scratch/username/Proposal
python3 src/benchmark_suite.py
```

This will take ~3 hours on a single GPU.

## After Completion
Download the results back to your Mac:
```bash
scp username@cluster.university.edu:/scratch/username/Proposal/benchmark_results.json \
  /Users/studio/Desktop/PhD/Proposal/
```

Then I can update the production paper with the final verified metrics.
