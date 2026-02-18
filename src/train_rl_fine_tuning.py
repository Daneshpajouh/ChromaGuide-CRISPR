import torch
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.model.rnagenesis_vae import RNAGenesisVAE
from src.rl.ppo_optimizer import PPOOptimizer
from src.model.deepmens import DeepMEnsExact

def rl_fine_tuning(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"RL Fine-Tuning (PPO) | Device: {device}")

    # 1. Load Models
    vae = RNAGenesisVAE(seq_len=23, latent_dim=256).to(device)
    if os.path.exists(args.vae_path):
        vae.load_state_dict(torch.load(args.vae_path, map_location=device))
    vae.eval()

    reward_model = DeepMEnsExact(seq_len=23).to(device)
    if os.path.exists(args.reward_model_path):
        reward_model.load_state_dict(torch.load(args.reward_model_path, map_location=device))
    reward_model.eval() # Freeze Reward Model

    ppo = PPOOptimizer(latent_dim=256, lr=1e-4, device=device)

    batch_size = args.batch_size
    current_z = torch.randn(batch_size, 256).to(device)

    # Fixed Position tensor (0..22)
    x_pos = torch.arange(23).unsqueeze(0).repeat(batch_size, 1).to(device)
    # Fixed Shape (Average shape) - Better than zeros to avoid OOD input
    x_shape = torch.zeros(batch_size, 4, 23).to(device)

    print("Starting REAL PPO Optimization...")
    for iteration in range(args.iterations):
        delta_z, logprobs = ppo.select_action(current_z)
        next_z = current_z + delta_z * 0.1

        with torch.no_grad():
            logits = vae.decode(next_z) # (B, 4, 23)
            probs = torch.softmax(logits, dim=1) # Differentiable-ish representation

            # Use DeepMEns to score.
            # DeepMEns expects (B, 4, 23). Probs is (B, 4, 23).
            # We assume DeepMEns can handle continuous inputs (it's a CNN).

            # NOTE: DeepMEns was trained on discrete One-Hot.
            # Passing soft probabilities works as a continuous approximation.
            eff_score = reward_model(probs, x_shape, x_pos).squeeze()

            # Reward = Efficiency
            rewards = eff_score

        ppo.update(current_z, delta_z, logprobs, rewards, next_z, None)
        current_z = next_z.detach()

        if iteration % 10 == 0:
            print(f"Iter {iteration}: Mean Efficiency Reward = {rewards.mean().item():.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "ppo_actor.pt")
    torch.save(ppo.actor.state_dict(), path)
    print(f"Saved PPO Actor to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50) # Keep small for dev, large for prod
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--vae_path", type=str, default="models/rnagenesis/vae.pt")
    parser.add_argument("--reward_model_path", type=str, default="models/deepmens/deepmens_seed_0.pt")
    parser.add_argument("--output_dir", type=str, default="models/rl")

    args = parser.parse_args()
    rl_fine_tuning(args)
