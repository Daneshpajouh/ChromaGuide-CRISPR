#!/usr/bin/env python3
"""Generate SLURM job scripts for all ChromaGuide experiments.

Creates SLURM scripts for all backbone × split × seed combinations,
with cluster-specific configurations.

Usage:
    python generate_slurm_jobs.py --output-dir slurm_jobs/
"""
import os
import sys
import argparse
from pathlib import Path

# ============================================================
# Cluster configurations
# ============================================================

CLUSTERS = {
    'narval': {
        'gpu_type': 'a100',
        'gpu_mem': '40G',
        'module_loads': 'module load StdEnv/2023 python/3.11.5 scipy-stack cuda/12.2',
        'account_gpu': 'def-kwiese_gpu',
        'account_cpu': 'def-kwiese_cpu',
        'max_time': '24:00:00',
        'mem': '64G',
    },
    'beluga': {
        'gpu_type': 'v100',
        'gpu_mem': '16G',
        'module_loads': 'module load StdEnv/2023 python/3.11.5 scipy-stack cuda/12.2',
        'account_gpu': 'def-kwiese_gpu',
        'account_cpu': 'def-kwiese_cpu',
        'max_time': '24:00:00',
        'mem': '32G',  # V100 nodes have less RAM
    },
    'rorqual': {
        'gpu_type': 'h100',
        'gpu_mem': '80G',
        'module_loads': 'module load StdEnv/2023 python/3.11.5 scipy-stack cuda/12.2',
        'account_gpu': 'def-kwiese_gpu',
        'account_cpu': 'def-kwiese_cpu',
        'max_time': '24:00:00',
        'mem': '64G',
    },
    'fir': {
        'gpu_type': 'a100',
        'gpu_mem': '80G',
        'module_loads': '# Fir: No module command, using system packages',
        'account_gpu': 'def-kwiese_gpu',
        'account_cpu': 'def-kwiese_cpu',
        'max_time': '24:00:00',
        'mem': '64G',
    },
    'killarney': {
        'gpu_type': 'h100',
        'gpu_mem': '80G',
        'module_loads': 'module load StdEnv/2023 python/3.11.5 scipy-stack cuda/12.2',
        'account_gpu': 'def-kwiese_gpu',
        'account_cpu': 'def-kwiese_cpu',
        'max_time': '24:00:00',
        'mem': '64G',
    },
    'nibi': {
        'gpu_type': 'gpu',
        'gpu_mem': '16G',
        'module_loads': 'module load StdEnv/2023 python/3.11.5 scipy-stack cuda/12.2',
        'account_gpu': 'def-kwiese_gpu',
        'account_cpu': 'def-kwiese_cpu',
        'max_time': '24:00:00',
        'mem': '32G',
    },
}

# ============================================================
# Experiment configurations
# ============================================================

# Backbone → GPU requirement mapping
BACKBONE_GPU_REQS = {
    'cnn_gru': {'min_vram': 4, 'time': '4:00:00', 'epochs': 100, 'batch_size': 256, 'patience': 15},
    'caduceus': {'min_vram': 8, 'time': '6:00:00', 'epochs': 100, 'batch_size': 256, 'patience': 15},
    'dnabert2': {'min_vram': 24, 'time': '12:00:00', 'epochs': 50, 'batch_size': 64, 'patience': 10},
    'nucleotide_transformer': {'min_vram': 48, 'time': '18:00:00', 'epochs': 30, 'batch_size': 16, 'patience': 10},
    'evo': {'min_vram': 16, 'time': '12:00:00', 'epochs': 30, 'batch_size': 32, 'patience': 10},
}

# Strategic cluster assignment for maximum parallelism
# Each backbone gets assigned to specific clusters based on GPU capability
CLUSTER_ASSIGNMENTS = {
    # CNN-GRU: lightweight, runs anywhere → spread across clusters
    'cnn_gru': ['narval', 'beluga', 'nibi'],
    # Caduceus: moderate → medium GPU clusters
    'caduceus': ['narval', 'killarney', 'beluga'],
    # DNABERT-2: needs 24GB+ → A100/H100 clusters
    'dnabert2': ['rorqual', 'killarney'],
    # Nucleotide Transformer: 500M params, needs 48GB+ → only 80GB GPUs
    'nucleotide_transformer': ['fir', 'rorqual'],
    # Evo: moderate with LoRA → A100/H100 clusters
    'evo': ['fir', 'killarney'],
}

# Seeds for each experiment (3 seeds for robustness)
SEEDS = [42, 123, 456]

# Splits to run
SPLITS = {
    'A': {'fold': 0},   # Gene-held-out (primary)
    'B': {'fold': 0},   # Dataset-held-out
    'C': {'fold': 0},   # Cell-line-held-out
}


def generate_slurm_script(backbone, split, split_fold, seed, cluster, output_dir):
    """Generate a SLURM job script."""
    cluster_cfg = CLUSTERS[cluster]
    backbone_cfg = BACKBONE_GPU_REQS[backbone]
    
    exp_name = f"{backbone}_split{split}_seed{seed}"
    job_name = f"cg-{backbone[:6]}-{split}-s{seed}"
    
    # Determine extra args based on backbone
    extra_args = []
    if backbone_cfg.get('epochs'):
        extra_args.append(f"--epochs {backbone_cfg['epochs']}")
    if backbone_cfg.get('batch_size'):
        extra_args.append(f"--batch-size {backbone_cfg['batch_size']}")
    
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={cluster_cfg['account_gpu']}
#SBATCH --time={backbone_cfg['time']}
#SBATCH --mem={cluster_cfg['mem']}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=results/{exp_name}/slurm-%j.out
#SBATCH --error=results/{exp_name}/slurm-%j.err

# ============================================================
# ChromaGuide: {backbone} | Split {split} | Seed {seed}
# Cluster: {cluster} ({cluster_cfg['gpu_type'].upper()})
# ============================================================

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Backbone: {backbone}"
echo "Split: {split}"
echo "Seed: {seed}"
echo "Cluster: {cluster}"
echo "==================="

# Load modules
{cluster_cfg['module_loads']}

# Activate virtual environment
source ~/scratch/chromaguide_v2_env/bin/activate

# Environment variables
export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=~/scratch/.cache/huggingface
export HF_HOME=~/scratch/.cache/huggingface
export TORCH_HOME=~/scratch/.cache/torch
export WANDB_MODE=offline
export OMP_NUM_THREADS=4

# Navigate to project
cd ~/scratch/chromaguide_v2

# Create output directory
mkdir -p results/{exp_name}

echo ""
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
python -c 'import torch; print(f"GPU: {{torch.cuda.get_device_name(0)}}") if torch.cuda.is_available() else print("No GPU")'
echo ""

# Run training
python experiments/train_experiment.py \\
    --backbone {backbone} \\
    --split {split} \\
    --split-fold {split_fold} \\
    --seed {seed} \\
    --data-dir ~/scratch/chromaguide_v2/data \\
    --output-dir ~/scratch/chromaguide_v2/results \\
    --no-wandb \\
    --patience {backbone_cfg['patience']} \\
    --gradient-clip 1.0 \\
    {' '.join(extra_args)}

EXIT_CODE=$?

echo ""
echo "===== JOB COMPLETE ====="
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

if [ -f results/{exp_name}/results.json ]; then
    echo ""
    echo "=== RESULTS ==="
    python -c "
import json
with open('results/{exp_name}/results.json') as f:
    r = json.load(f)
print(f'Spearman: {{r[\"test_metrics\"][\"spearman\"]:.4f}}')
print(f'Pearson:  {{r[\"test_metrics\"][\"pearson\"]:.4f}}')
print(f'ECE:      {{r[\"test_metrics\"][\"ece\"]:.4f}}')
print(f'Coverage: {{r[\"conformal\"][\"coverage\"]:.4f}}')
print(f'Time:     {{r[\"training_time_seconds\"]/60:.1f}} min')
"
fi

exit $EXIT_CODE
"""
    
    # Write script
    script_name = f"{exp_name}_{cluster}.sh"
    script_path = output_dir / script_name
    script_path.write_text(script)
    os.chmod(str(script_path), 0o755)
    
    return script_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='slurm_jobs')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all job scripts
    jobs = []  # (backbone, split, seed, cluster, script_name)
    
    for backbone, clusters in CLUSTER_ASSIGNMENTS.items():
        for split, split_cfg in SPLITS.items():
            for seed_idx, seed in enumerate(SEEDS):
                # Assign to cluster in round-robin
                cluster = clusters[seed_idx % len(clusters)]
                
                script_name = generate_slurm_script(
                    backbone=backbone,
                    split=split,
                    split_fold=split_cfg['fold'],
                    seed=seed,
                    cluster=cluster,
                    output_dir=output_dir,
                )
                
                jobs.append({
                    'backbone': backbone,
                    'split': split,
                    'seed': seed,
                    'cluster': cluster,
                    'script': script_name,
                })
    
    # Generate submission script per cluster
    cluster_jobs = {}
    for job in jobs:
        cluster = job['cluster']
        if cluster not in cluster_jobs:
            cluster_jobs[cluster] = []
        cluster_jobs[cluster].append(job)
    
    for cluster, cjobs in cluster_jobs.items():
        submit_script = f"""#!/bin/bash
# Submit all ChromaGuide jobs on {cluster}
# Generated automatically

echo "Submitting {len(cjobs)} jobs on {cluster}..."
cd ~/scratch/chromaguide_v2

"""
        for job in cjobs:
            submit_script += f"""echo "Submitting: {job['backbone']} | Split {job['split']} | Seed {job['seed']}"
sbatch experiments/slurm_jobs/{job['script']}

"""
        
        submit_script += f"""
echo ""
echo "All {len(cjobs)} jobs submitted on {cluster}."
echo "Check status: squeue -u $USER"
"""
        
        submit_path = output_dir / f"submit_{cluster}.sh"
        submit_path.write_text(submit_script)
        os.chmod(str(submit_path), 0o755)
    
    # Summary
    print("=" * 70)
    print("SLURM Job Generation Summary")
    print("=" * 70)
    print(f"\nTotal jobs: {len(jobs)}")
    print(f"Output dir: {output_dir}")
    print()
    
    print("Jobs per cluster:")
    for cluster, cjobs in sorted(cluster_jobs.items()):
        print(f"  {cluster}: {len(cjobs)} jobs")
        for job in cjobs:
            print(f"    - {job['backbone']} | Split {job['split']} | Seed {job['seed']}")
    
    print()
    print("Jobs per backbone:")
    backbone_counts = {}
    for job in jobs:
        backbone_counts.setdefault(job['backbone'], []).append(job)
    for backbone, bjobs in sorted(backbone_counts.items()):
        print(f"  {backbone}: {len(bjobs)} jobs")
    
    print()
    print("Submission commands (per cluster):")
    for cluster in sorted(cluster_jobs.keys()):
        print(f"  ssh {cluster} 'bash ~/scratch/chromaguide_v2/experiments/slurm_jobs/submit_{cluster}.sh'")


if __name__ == '__main__':
    main()
