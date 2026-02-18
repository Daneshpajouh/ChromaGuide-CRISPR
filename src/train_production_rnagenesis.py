import os
import subprocess
import argparse

def train_rnagenesis_production():
    print("🚀 LAUNCHING PRODUCTION RNAGENESIS TRAINING... 🚀")

    # 1. Train VAE
    print("\n--- Phase 1: VAE Representation Learning ---")
    subprocess.run([
        "python3", "src/train_rnagenesis_vae.py",
        "--epochs", "5",
        "--batch_size", "64",
        "--output_dir", "models/rnagenesis/prod"
    ], check=True)

    # 2. Train Diffusion
    print("\n--- Phase 2: Latent Diffusion Training ---")
    subprocess.run([
        "python3", "src/train_rnagenesis_diffusion.py",
        "--epochs", "20",
        "--batch_size", "64",
        "--vae_path", "models/rnagenesis/prod/vae.pt",
        "--output_dir", "models/rnagenesis/prod"
    ], check=True)

    # 3. RL Fine-Tuning (Real SOTA)
    print("\n--- Phase 3: RL-PPO Fine-Tuning ---")
    # Using Model Seed 0 as the Reward Model
    reward_model_path = "models/deepmens/deepmens_seed_0.pt"

    if os.path.exists(reward_model_path):
        subprocess.run([
            "python3", "src/train_rl_fine_tuning.py",
            "--iterations", "50", # Real optimization steps
            "--batch_size", "64",
            "--vae_path", "models/rnagenesis/prod/vae.pt",
            "--reward_model_path", reward_model_path,
            "--output_dir", "models/rnagenesis/prod/rl"
        ], check=True)
    else:
        print(f"⚠️ Warning: Reward model not found at {reward_model_path}. Skipping RL.")

    print("✅ RNAGenesis Production Pipeline Complete.")

if __name__ == "__main__":
    train_rnagenesis_production()
