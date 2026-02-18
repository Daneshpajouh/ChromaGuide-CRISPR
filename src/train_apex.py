import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
from src.model.crispro_apex import CRISPRO_Apex
from src.model.geometric_optimizer import GeometricCRISPROptimizer

class ApexSequenceDataset(Dataset):
    def __init__(self, csv_path, seq_len=256):
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[idx]
        seq_str = str(row['sequence']).upper()
        encoded = [self.vocab.get(char, 4) for char in seq_str[:self.seq_len]]
        if len(encoded) < self.seq_len:
            encoded += [4] * (self.seq_len - len(encoded))
        dna_tensor = torch.tensor(encoded, dtype=torch.long)
        epi_tensor = torch.zeros((self.seq_len, 5), dtype=torch.float32)
        efficiency = float(row['efficiency'])
        return {
            'dna': dna_tensor,
            'epi': epi_tensor,
            'target': torch.tensor(efficiency, dtype=torch.float32)
        }

def train_apex(args):
    print(f"🔥 CRISPRO-APEX: {args.job_name} 🔥")
    print("=" * 60)

    # 1. Device Setup
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"[*] Training on: {device}")

    # 2. Data
    full_dataset = ApexSequenceDataset(args.data_path, seq_len=args.seq_len)

    if args.subset_size > 0:
        indices = torch.randperm(len(full_dataset))[:args.subset_size]
        subset = torch.utils.data.Subset(full_dataset, indices)
    else:
        subset = full_dataset

    train_size = int(0.95 * len(subset))
    val_size = len(subset) - train_size
    train_ds, val_ds = random_split(subset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"[*] Training Samples: {train_size:,}")
    print(f"[*] Validation Samples: {val_size:,}")

    # 3. Model
    model = CRISPRO_Apex(d_model=256, n_layers=4, n_modalities=5, chunk_size=64).to(device)

    # 4. Optimizers
    base_optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    geometric_opt = GeometricCRISPROptimizer(model, lr=0.01, damping=1.0)

    criterion = nn.MSELoss()

    # 5. Training Loop
    best_scc = -1.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for i, batch in enumerate(pbar):
            dna = batch['dna'].to(device)
            epi = batch['epi'].to(device)
            target = batch['target'].to(device).view(-1, 1)

            base_optimizer.zero_grad()
            out = model(dna, epi)
            loss = criterion(out['on_target'], target)
            loss.backward()
            base_optimizer.step()

            if i % 100 == 0:
                 geometric_opt.step(dna, target)

            total_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                dna = batch['dna'].to(device)
                epi = batch['epi'].to(device)
                target = batch['target'].to(device)
                out = model(dna, epi)
                preds.extend(out['on_target'].cpu().flatten().numpy())
                targets.extend(target.cpu().flatten().numpy())

        scc, _ = spearmanr(targets, preds)
        print(f"[*] Epoch {epoch+1} Valid SCC: {scc:.4f}")

        if scc > best_scc:
            best_scc = scc
            ckpt_path = f"checkpoints/{args.job_name}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'scc': scc,
            }, ckpt_path)
            print(f"🌟 Progress: Best SCC Updated to {best_scc:.4f}")

    print("=" * 60)
    print(f"🏆 FINAL AUTHENTIC SPEARMAN SCC: {best_scc:.4f}")
    print(f"📊 SOTA TARGET (CRISPR_HNN): 0.891")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_name", type=str, default="apex_real_world")
    parser.add_argument("--data_path", type=str, default="merged_crispr_data.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--subset_size", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    train_apex(args)
