#!/usr/bin/env python3
"""Generate SLURM scripts for ChromaGuide v4 experiments."""

import os
from pathlib import Path

BACKBONES = ['cnn_gru', 'caduceus', 'dnabert2', 'evo', 'nucleotide_transformer']
SPLITS = ['A', 'B', 'C']
SEEDS = [42, 123, 456]

# Cluster configs
CLUSTERS = {
    'nibi': {
        'account': 'def-kwiese_gpu',
        'partition': None,
        'gpu_flag': '--gres=gpu:1',
        'module_cmd': 'module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null',
        'source_cmd': '',
    },
    'narval': {
        'account': 'def-kwiese_gpu',
        'partition': None,
        'gpu_flag': '--gres=gpu:1',
        'module_cmd': 'module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null',
        'source_cmd': '',
    },
    'rorqual': {
        'account': 'def-kwiese_gpu',
        'partition': None,
        'gpu_flag': '--gres=gpu:1',
        'module_cmd': 'module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null',
        'source_cmd': '',
    },
    'fir': {
        'account': 'def-kwiese_gpu',
        'partition': 'gpubase_bygpu_b3',
        'gpu_flag': '--gres=gpu:1',
        'module_cmd': '',
        'source_cmd': 'source /cvmfs/soft.computecanada.ca/config/profile/bash.sh\nmodule load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null',
    },
}

# Resource allocation per backbone
RESOURCES = {
    'cnn_gru':                {'mem': '32G', 'time': '8:00:00',  'cpus': 4},
    'caduceus':               {'mem': '48G', 'time': '10:00:00', 'cpus': 4},
    'dnabert2':               {'mem': '64G', 'time': '10:00:00', 'cpus': 4},
    'evo':                    {'mem': '64G', 'time': '12:00:00', 'cpus': 4},
    'nucleotide_transformer': {'mem': '64G', 'time': '12:00:00', 'cpus': 4},
}


def generate_slurm_script(backbone, split, seed, cluster):
    """Generate a SLURM script for one experiment."""
    res = RESOURCES[backbone]
    cc = CLUSTERS[cluster]
    
    job_name = f"cg4_{backbone[:4]}_{split}_s{seed}"
    exp_name = f"cg4_{backbone}_s{split}_f0_r{seed}"
    
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={cc['account']}",
        f"#SBATCH {cc['gpu_flag']}",
        f"#SBATCH --cpus-per-task={res['cpus']}",
        f"#SBATCH --mem={res['mem']}",
        f"#SBATCH --time={res['time']}",
    ]
    
    if cc['partition']:
        lines.append(f"#SBATCH --partition={cc['partition']}")
    
    lines.extend([
        f"",
        f"#SBATCH --output=~/scratch/chromaguide_v4/logs/{exp_name}_%j.out",
        f"#SBATCH --error=~/scratch/chromaguide_v4/logs/{exp_name}_%j.err",
        f"",
        f'echo "============================================"',
        f'echo "ChromaGuide v4 Experiment"',
        f'echo "Job: {exp_name}"',
        f'echo "Cluster: {cluster}"',
        f'echo "Backbone: {backbone}"',
        f'echo "Split: {split} (fold 0)"',
        f'echo "Seed: {seed}"',
        f'echo "Date: $(date)"',
        f'echo "Node: $(hostname)"',
        f'echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"',
        f'echo "============================================"',
        f"",
        f"# Source environment",
    ])
    
    if cc['source_cmd']:
        lines.append(cc['source_cmd'])
    if cc['module_cmd']:
        lines.append(cc['module_cmd'])
    
    lines.extend([
        f"source ~/scratch/chromaguide_v2_env/bin/activate",
        f"",
        f"# Ensure dependencies",
        f"pip install --no-index --upgrade pip 2>/dev/null",
        f"pip install omegaconf scipy 2>/dev/null",
        f"",
        f"# Set environment variables",
        f"export TRANSFORMERS_CACHE=~/scratch/chromaguide_v4/model_cache",
        f"export HF_HOME=~/scratch/chromaguide_v4/model_cache",
        f"export TORCH_HOME=~/scratch/chromaguide_v4/model_cache",
        f"export TOKENIZERS_PARALLELISM=false",
        f"export OMP_NUM_THREADS={res['cpus']}",
        f"",
        f"# Create directories",
        f"mkdir -p ~/scratch/chromaguide_v4/logs",
        f"mkdir -p ~/scratch/chromaguide_v4/results_v4",
        f"mkdir -p ~/scratch/chromaguide_v4/model_cache",
        f"",
        f"cd ~/scratch/chromaguide_v4",
        f"",
        f"# Run v4 experiment",
        f'echo "Starting v4 training..."',
        f"python experiments/train_experiment_v4.py \\",
        f"    --backbone {backbone} \\",
        f"    --split {split} \\",
        f"    --split-fold 0 \\",
        f"    --seed {seed} \\",
        f"    --data-dir ~/scratch/chromaguide_v4/data \\",
        f"    --output-dir ~/scratch/chromaguide_v4/results_v4 \\",
        f"    --patience 30 \\",
        f"    --gradient-clip 1.0 \\",
        f"    --mixed-precision \\",
        f"    --swa \\",
        f"    --mixup-alpha 0.2 \\",
        f"    --rc-augment \\",
        f"    --label-smoothing 0.01 \\",
        f"    --no-wandb \\",
        f"    --model-cache-dir ~/scratch/chromaguide_v4/model_cache \\",
        f"    --version v4",
        f"",
        f'echo "============================================"',
        f'echo "Experiment complete: {exp_name}"',
        f'echo "Date: $(date)"',
        f'echo "============================================"',
    ])
    
    return '\n'.join(lines) + '\n'


def main():
    script_dir = Path(__file__).resolve().parent
    
    for cluster in CLUSTERS:
        out_dir = script_dir / f'slurm_v4_{cluster}'
        out_dir.mkdir(exist_ok=True)
        
        submit_lines = ["#!/bin/bash", f"# Submit all 45 ChromaGuide v4 experiments on {cluster}", ""]
        
        count = 0
        for backbone in BACKBONES:
            for split in SPLITS:
                for seed in SEEDS:
                    script = generate_slurm_script(backbone, split, seed, cluster)
                    filename = f"cg4_{backbone}_s{split}_f0_r{seed}.sh"
                    filepath = out_dir / filename
                    filepath.write_text(script)
                    
                    submit_lines.append(f"sbatch {filename}")
                    submit_lines.append("sleep 1  # Avoid rate limiting")
                    count += 1
        
        submit_lines.append(f"\necho 'Submitted {count} jobs'")
        (out_dir / 'submit_all.sh').write_text('\n'.join(submit_lines) + '\n')
        
        print(f"{cluster}: Generated {count} SLURM scripts in {out_dir}/")


if __name__ == '__main__':
    main()
