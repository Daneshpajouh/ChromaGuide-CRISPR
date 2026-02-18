import os
import subprocess
import argparse

def train_ensemble_production():
    print("🚀 LAUNCHING PRODUCTION DEEPMENS ENSEMBLE TRAINING (REAL DATA) 🚀")
    print("This will train 5 models on the FULL dataset. Please wait...")

    seeds = [0, 1, 2, 3, 4]
    processes = []

    # Sequential training to avoid killing the Mac CPU/RAM
    for seed in seeds:
        print(f"\nTraining Model {seed+1}/5 (Seed {seed})...")
        cmd = [
            "python3", "src/train_deepmens.py",
            "--seed", str(seed),
            "--epochs", "5", # 5 Epochs on full data is substantial
            "--batch_size", "64",
            "--output_dir", "models/deepmens"
            # NO --use_mini flag!
        ]

        subprocess.run(cmd, check=True)
        print(f"Model {seed} Complete.")

    print("\n✅ Ensemble Training Optimization Complete.")

if __name__ == "__main__":
    train_ensemble_production()
