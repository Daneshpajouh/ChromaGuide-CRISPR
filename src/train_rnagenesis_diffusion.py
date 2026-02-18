import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.model.rnagenesis_vae import RNAGenesisVAE
from src.model.rnagenesis_diffusion import RNAGenesisDiffusion, DiffusionSchedule
from src.data.crisprofft import CRISPRoffTDataset
from src.train_deepmens import DeepMEnsDatasetWrapper

def train_diffusion(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"RNAGenesis Diffusion Training | Device: {device}")

    # 1. Load Pre-trained VAE
    vae = RNAGenesisVAE(seq_len=23, latent_dim=256).to(device)
    if os.path.exists(args.vae_path):
        vae.load_state_dict(torch.load(args.vae_path, map_location=device))
        print("Loaded VAE.")
    else:
        print("⚠️ VAE path invalid. Using random VAE (Testing only).")

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # 2. Diffusion Model
    diffusion = RNAGenesisDiffusion(latent_dim=256).to(device)
    optimizer = optim.Adam(diffusion.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    schedule = DiffusionSchedule(T=1000)

    # 3. Data
    train_base = CRISPRoffTDataset(split='train', use_mini=args.use_mini)
    train_dataset = DeepMEnsDatasetWrapper(train_base)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        diffusion.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for seq, _, _, _ in loop:
            seq = seq.to(device)

            # Encode to Latent
            with torch.no_grad():
                mu, logvar = vae.encode(seq)
                z_0 = vae.reparameterize(mu, logvar) # (B, 256)

            # Sample Time t
            t = schedule.sample_timesteps(seq.shape[0], device) # (B,)

            # Add Noise
            z_t, epsilon = schedule.noise_image(z_0, t)

            # Context (Gene ID) - Mock random for now
            # In production, dataset should return gene_id
            gene_ids = torch.randint(0, 20000, (seq.shape[0],), device=device)

            # Predict Noise
            # t needs to be float or embedding input? My model takes float or long?
            # Model takes t and passes to embedding. It expects input for embedding layer?
            # self.time_emb = nn.Sequential(nn.Linear(1, 128)...)
            # So it expects float (B, 1). Wait, I implemented nn.Linear(1, ...)
            # So I should pass t.float().unsqueeze(-1) normalized?
            # Let's fix input: t is long [0, 999].
            # If model uses nn.Linear(1, ...), I should pass t/T as float.

            t_norm = t.float().unsqueeze(-1) / 1000.0
            eps_pred = diffusion(z_t, t_norm, gene_ids)

            loss = criterion(eps_pred, epsilon)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Avg Loss: {train_loss/len(train_loader):.4f}")

    path = os.path.join(args.output_dir, "diffusion.pt")
    torch.save(diffusion.state_dict(), path)
    print(f"Saved Diffusion to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_mini", action="store_true")
    parser.add_argument("--vae_path", type=str, default="models/rnagenesis/vae.pt")
    parser.add_argument("--output_dir", type=str, default="models/rnagenesis")

    args = parser.parse_args()
    train_diffusion(args)
