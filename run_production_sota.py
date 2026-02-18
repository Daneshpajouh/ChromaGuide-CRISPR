import os
import subprocess
import time

def run_step(cmd, desc):
    print(f'--- {desc} ---')
    print(f'Command: {" ".join(cmd)}')
    start = time.time()
    subprocess.run(cmd, check=True)
    end = time.time()
    print(f'Done in {end-start:.2f}s\n')

def main():
    # Production Path on Nibi
    os.chdir('/home/amird/projects/def-kwiese/amird/Proposal')
    os.makedirs('models/deepmens', exist_ok=True)
    os.makedirs('models/rnagenesis', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # 1. Train Ensemble (Seeds 0-4)
    # We use 5 epochs for speed during this resurgence sprint
    for seed in range(5):
        run_step([
            'python3', 'src/train_deepmens.py',
            '--seed', str(seed),
            '--epochs', '5',
            '--batch_size', '128'
        ], f'Training DeepMEns Model {seed}')

    # 2. Train RNAGenesis VAE
    run_step([
        'python3', 'src/train_rnagenesis_vae.py',
        '--epochs', '10',
        '--batch_size', '128'
    ], 'Training RNAGenesis VAE')

    # 3. Train RNAGenesis Diffusion
    run_step([
        'python3', 'src/train_rnagenesis_diffusion.py',
        '--epochs', '10',
        '--batch_size', '128',
        '--vae_path', 'models/rnagenesis/vae.pt'
    ], 'Training RNAGenesis Diffusion')

    # 4. Run Final Benchmark
    run_step(['python3', 'src/benchmark_suite.py'], 'Running Final SOTA Benchmark')

if __name__ == '__main__':
    main()
