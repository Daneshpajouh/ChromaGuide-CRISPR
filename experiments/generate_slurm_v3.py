#!/usr/bin/env python3
"""Generate SLURM job scripts for ChromaGuide v3 experiments.

Generates jobs for ALL combinations of:
    - 5 backbones: cnn_gru, dnabert2, caduceus, evo, nucleotide_transformer
    - 3 splits: A, B(fold=0), C(fold=0)
    - 3 seeds: 42, 123, 456

Total: 5 × 3 × 3 = 45 experiments per cluster
"""
import os
import sys
from pathlib import Path

BACKBONES = ['cnn_gru', 'caduceus', 'dnabert2', 'evo', 'nucleotide_transformer']
SPLITS = [('A', 0), ('B', 0), ('C', 0)]
SEEDS = [42, 123, 456]

# Time limits per backbone (conservative)
TIME_LIMITS = {
    'cnn_gru': '6:00:00',
    'caduceus': '8:00:00',
    'dnabert2': '12:00:00',
    'evo': '12:00:00',
    'nucleotide_transformer': '18:00:00',
}

# Memory per backbone
MEMORY = {
    'cnn_gru': '32G',
    'caduceus': '32G',
    'dnabert2': '64G',
    'evo': '64G',
    'nucleotide_transformer': '96G',
}

def generate_slurm_script(
    backbone: str,
    split: str,
    split_fold: int,
    seed: int,
    cluster: str = 'nibi',
    account: str = 'def-kwiese_gpu',
    partition: str = None,
    scratch_dir: str = '~/scratch/chromaguide_v3',
    venv_dir: str = '~/scratch/chromaguide_v2_env',
) -> str:
    """Generate a single SLURM script."""
    job_name = f"cg3_{backbone[:4]}_s{split}_f{split_fold}_r{seed}"
    time_limit = TIME_LIMITS.get(backbone, '12:00:00')
    mem = MEMORY.get(backbone, '64G')
    
    # Partition handling
    partition_line = ""
    if cluster == 'fir':
        # Fir: use gpubase_bygpu_b2 for ≤12h, b3 for ≤24h
        hours = int(time_limit.split(':')[0])
        if hours <= 12:
            partition_line = "#SBATCH --partition=gpubase_bygpu_b2"
        else:
            partition_line = "#SBATCH --partition=gpubase_bygpu_b3"
    elif partition:
        partition_line = f"#SBATCH --partition={partition}"
    
    # Model cache directory (for pretrained models)
    model_cache = f"{scratch_dir}/model_cache"
    
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
{partition_line}
#SBATCH --output={scratch_dir}/logs/{job_name}_%j.out
#SBATCH --error={scratch_dir}/logs/{job_name}_%j.err

echo "============================================"
echo "ChromaGuide v3 Experiment"
echo "Job: {job_name}"
echo "Cluster: {cluster}"
echo "Backbone: {backbone}"
echo "Split: {split} (fold {split_fold})"
echo "Seed: {seed}"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# Source environment
"""
    
    if cluster == 'fir':
        script += "source /cvmfs/soft.computecanada.ca/config/profile/bash.sh\n"
    
    script += f"""
module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null
source {venv_dir}/bin/activate

# Ensure dependencies
pip install --no-index --upgrade pip 2>/dev/null
pip install omegaconf scipy 2>/dev/null

# Set environment variables
export TRANSFORMERS_CACHE={model_cache}
export HF_HOME={model_cache}
export TORCH_HOME={model_cache}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# Create directories
mkdir -p {scratch_dir}/logs
mkdir -p {scratch_dir}/results_v3
mkdir -p {model_cache}

cd {scratch_dir}

# Run experiment
echo "Starting training..."
python experiments/train_experiment_v3.py \\
    --backbone {backbone} \\
    --split {split} \\
    --split-fold {split_fold} \\
    --seed {seed} \\
    --data-dir {scratch_dir}/data \\
    --output-dir {scratch_dir}/results_v3 \\
    --loss-type logcosh \\
    --optimizer adamax \\
    --lambda-rank 0.1 \\
    --patience 20 \\
    --gradient-clip 1.0 \\
    --mixed-precision \\
    --no-wandb \\
    --model-cache-dir {model_cache} \\
    --version v3

echo "============================================"
echo "Experiment complete: {job_name}"
echo "Date: $(date)"
echo "============================================"
"""
    return script


def generate_all_jobs(output_dir: str, cluster: str = 'nibi', **kwargs):
    """Generate all 45 job scripts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scripts = []
    for backbone in BACKBONES:
        for split, fold in SPLITS:
            for seed in SEEDS:
                script = generate_slurm_script(
                    backbone=backbone,
                    split=split,
                    split_fold=fold,
                    seed=seed,
                    cluster=cluster,
                    **kwargs,
                )
                
                job_name = f"cg3_{backbone}_s{split}_f{fold}_r{seed}"
                script_path = output_dir / f"{job_name}.sh"
                with open(script_path, 'w') as f:
                    f.write(script)
                scripts.append(str(script_path))
    
    # Generate submission script
    submit_script = output_dir / "submit_all.sh"
    with open(submit_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Submit all {len(scripts)} ChromaGuide v3 experiments\n\n")
        for s in scripts:
            f.write(f"sbatch {Path(s).name}\n")
            f.write("sleep 1  # Avoid rate limiting\n")
        f.write(f"\necho 'Submitted {len(scripts)} jobs'\n")
    os.chmod(submit_script, 0o755)
    
    return scripts


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='slurm_v3_jobs')
    parser.add_argument('--cluster', type=str, default='nibi',
                        choices=['nibi', 'narval', 'rorqual', 'fir'])
    parser.add_argument('--account', type=str, default='def-kwiese_gpu')
    parser.add_argument('--scratch-dir', type=str, default='~/scratch/chromaguide_v3')
    parser.add_argument('--venv-dir', type=str, default='~/scratch/chromaguide_v2_env')
    args = parser.parse_args()
    
    scripts = generate_all_jobs(
        args.output_dir,
        cluster=args.cluster,
        account=args.account,
        scratch_dir=args.scratch_dir,
        venv_dir=args.venv_dir,
    )
    print(f"Generated {len(scripts)} SLURM job scripts in {args.output_dir}/")
