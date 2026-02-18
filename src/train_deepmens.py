import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
from scipy.stats import spearmanr

# Add Project Root to Path
sys.path.append(os.getcwd())

# Project Imports
from src.model.deepmens import DeepMEnsExact
from src.data.crisprofft import CRISPRoffTDataset
from src.data.dna_shape import shape_featurizer

# ----------------------------------------
# 1. Dataset Wrapper for DeepMEns
# ----------------------------------------
class DeepMEnsDatasetWrapper(Dataset):
    """
    Wraps CRISPRoffTDataset to provide the 3-branch inputs:
    1. Sequence (One-Hot) -> (4, L)
    2. Shape (MGW, ProT...) -> (4, L)
    3. Positions -> (L,)
    """
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.vocab = base_dataset.vocab

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # Base returns: (seq_tensor, epigenetic_tensor, label)
        # Note: seq_tensor is (L,) integers from _tokenize

        # We need raw string to compute shape
        # But base dataset implementation tokenizes inside __getitem__ usually?
        # Let's check crisprofft.py: __getitem__ calls _tokenize

        # To avoid double processing, let's just reverse index or modify base.
        # Actually crisprofft.py's __getitem__ returns (seq_tensor, epi, val)
        # We need the string for shape.
        # We will access the dataframe directly from self.base.data

        row = self.base.data.iloc[idx]
        seq_str = row[self.base.seq_col]

        # Enforce Fixed Length (23bp) for DeepMEns (Spacer+PAM)
        target_len = 23
        if len(seq_str) > target_len:
            seq_str = seq_str[:target_len]
        elif len(seq_str) < target_len:
            seq_str = seq_str + 'N' * (target_len - len(seq_str))

        # 1. Sequence Branch (One-Hot)
        # Standard Mapping: A, C, G, T -> 0, 1, 2, 3
        local_vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
        seq_ints = torch.tensor([local_vocab.get(c, -1) for c in seq_str], dtype=torch.long)

        # Create (L, 4) one-hot. -1 will result in a zero-row.
        seq_onehot = torch.zeros(len(seq_str), 4)
        for i, idx in enumerate(seq_ints):
            if idx >= 0:
                seq_onehot[i, idx] = 1.0

        # Transpose to (4, L) for Conv1d
        seq_onehot = seq_onehot.permute(1, 0).float()

        # 2. Shape Branch
        shape_feat = shape_featurizer.get_shape(seq_str) # (4, L)

        # 3. Position Branch
        # Just integer positions [0, 1, ... L-1]
        # DeepMEns uses learnable embedding
        pos_ints = torch.arange(len(seq_str), dtype=torch.long)

        # 4. Label
        label_val = row[self.base.label_col]
        # Robust normalization for Spearman (preserve rank, avoid sigmoid saturation)
        if self.base.label_col == 'Score':
            # Use log-scale for high dynamic range scores
            label = torch.tensor(np.log1p(label_val) / np.log1p(1e6), dtype=torch.float32)
        else:
            # Assume percentage or probability
            label = torch.tensor(label_val / 100.0 if label_val > 1.0 else label_val, dtype=torch.float32)

        return seq_onehot, shape_feat, pos_ints, label.clamp(0, 1)

import torch.nn.functional as F

# ----------------------------------------
# 2. Training Loop
# ----------------------------------------

def train_deepmens_model(args):
    # Set Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DeepMEns Training | Seed: {args.seed} | Device: {device}")

    # Load Data
    # For Ensemble Diversity: We can subsample or just use different seeds for initialization/shuffle
    # The SOTA paper says "Data-Driven Ensemble" implies training on different splits.
    # We will use the full training set but shuffle differently (DataLoader does this).

    train_base = CRISPRoffTDataset(split='train', use_mini=args.use_mini)
    train_dataset = DeepMEnsDatasetWrapper(train_base)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = DeepMEnsExact(seq_len=23).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # SOTA: 1e-3 initial
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = nn.MSELoss()

    model_name = f"deepmens_seed_{args.seed}"
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
    parser.add_argument("--use_mini", action="store_true", help="Use mini dataset for debugging")
    parser.add_argument("--output_dir", type=str, default="models/deepmens", help="Output directory")

    args = parser.parse_args()

    train_deepmens_model(args)
