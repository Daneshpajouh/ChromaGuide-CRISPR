import argparse
import sys
import os
import torch
import numpy as np

# Add Project Root
sys.path.append(os.getcwd())

from src.train_deepmens import train_deepmens_model

class EnsembleArgs:
    """Mock arguments for function call"""
    def __init__(self, seed, epochs, batch_size, use_mini, output_dir):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_mini = use_mini
        self.output_dir = output_dir

def train_ensemble(epochs=20, batch_size=64, use_mini=False):
    print("="*60)
    print("🚀 STARTING DEEPMENS ENSEMBLE TRAINING (5 MODELS)")
    print("="*60)

    seeds = [0, 1, 2, 3, 4]
    output_dir = "models/deepmens"

    for seed in seeds:
        print(f"\nTraining Model {seed+1}/5 (Seed: {seed})...")
        args = EnsembleArgs(
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            use_mini=use_mini,
            output_dir=output_dir
        )
        try:
            train_deepmens_model(args)
            print(f"✅ Model {seed+1} Completed Successfully.")
        except Exception as e:
            print(f"❌ Model {seed+1} Failed: {e}")
            # Continue to next model? Yes.

    print("\n" + "="*60)
    print("🏁 ENSEMBLE TRAINING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--use_mini", action="store_true", help="Use mini dataset")

    args = parser.parse_args()

    train_ensemble(epochs=args.epochs, batch_size=args.batch_size, use_mini=args.use_mini)
