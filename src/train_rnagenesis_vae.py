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

from src.model.rnagenesis_vae import RNAGenesisVAE, vae_loss_function
from src.data.crisprofft import CRISPRoffTDataset
from src.train_deepmens import DeepMEnsDatasetWrapper

def train_vae(args):
    # Set Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"RNAGenesis VAE Training | Device: {device}")

    # Load Data
    train_base = CRISPRoffTDataset(split='train', use_mini=args.use_mini)
    train_dataset = DeepMEnsDatasetWrapper(train_base)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = RNAGenesisVAE(seq_len=23, latent_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for seq, _, _, _ in loop:
            # seq: (B, 4, 23)
            seq = seq.to(device)

            optimizer.zero_grad()
            recon_logits, mu, logvar = model(seq)

            loss = vae_loss_function(recon_logits, seq, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Avg Loss: {train_loss/len(train_loader):.4f}")

    path = os.path.join(args.output_dir, "vae.pt")
    torch.save(model.state_dict(), path)
    print(f"Saved VAE to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_mini", action="store_true")
    parser.add_argument("--output_dir", type=str, default="models/rnagenesis")

    args = parser.parse_args()
    train_vae(args)
