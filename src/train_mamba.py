import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.model.mamba_deepmens import MambaDeepMEns
from src.data.crisprofft import CRISPRoffTDataset
from src.train_deepmens import DeepMEnsDatasetWrapper

def train_mamba_model(args):
    # Set Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"MambaDeepMEns Training | Seed: {args.seed} | Device: {device}")

    # Load Data
    train_base = CRISPRoffTDataset(split='train', use_mini=args.use_mini)
    train_dataset = DeepMEnsDatasetWrapper(train_base)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Model: Mamba
    model = MambaDeepMEns(seq_len=23).to(device)

    # Optimizer (Mamba often likes lower LR or weight decay)
    # Using DeepMEns defaults for baseline comparison
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = nn.MSELoss()

    model_name = f"mamba_seed_{args.seed}"
    print(f"Starting Training: {model_name}")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for seq, shape, pos, label in loop:
            seq, shape, pos, label = seq.to(device), shape.to(device), pos.to(device), label.to(device)

            optimizer.zero_grad()
            pred = model(seq, shape, pos).squeeze()

            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        epoch_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss:.4f}")

        scheduler.step(epoch_loss)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--use_mini", action="store_true", help="Use mini dataset")
    parser.add_argument("--output_dir", type=str, default="models/mamba", help="Output directory")

    args = parser.parse_args()
    train_mamba_model(args)
