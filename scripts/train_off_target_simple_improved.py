#!/usr/bin/env python3
"""
Improved Off-target prediction with better class imbalance handling.
Target: AUROC >= 0.99

Key improvements over baseline:
1. Deeper/wider CNN with residual connections
2. Weighted sampling instead of SMOTE (faster, simpler)
3. Cosine annealing LR scheduler with warmup
4. Higher patience (30 epochs), max epochs (500)
5. Gradient accumulation
6. d_model = 64 as per thesis proposal
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Check device availability
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def one_hot_encode(seq, length=23):
    """One-hot encode DNA sequences"""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((4, length), dtype=np.float32)
    s = seq.upper()[:length]
    for j, c in enumerate(s):
        if j < length and c in mapping:
            encoded[mapping[c], j] = 1
    return encoded

def load_crisprofft_data(data_path, max_samples=None):
    """Load CRISPRoffT dataset"""
    seqs, labels = [], []
    print(f"Loading data from {data_path}...")

    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split('\t')
            if len(parts) < 35:
                continue
            try:
                guide_seq = parts[21]
                target_status = parts[33]

                if target_status not in ["ON", "OFF"]:
                    continue

                label = 1 if target_status == "OFF" else 0

                if not guide_seq or len(guide_seq) < 20:
                    continue

                seqs.append(guide_seq)
                labels.append(label)

                if max_samples and len(seqs) >= max_samples:
                    break
            except (IndexError, ValueError):
                continue

    print(f"Loaded {len(seqs)} sequences")
    X = np.array([one_hot_encode(seq) for seq in seqs]).astype(np.float32)
    y = np.array(labels).astype(np.float32)

    print(f"Data shape: {X.shape}")
    print(f"ON: {(y==0).sum()}, OFF: {(y==1).sum()}")
    print(f"Class imbalance: {(y==1).sum()/(y==0).sum():.1f}:1 OFF:ON")

    return X, y


class ResidualBlock(nn.Module):
    """Residual block"""
    def __init__(self, in_ch, out_ch, ks=5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, ks, padding=ks//2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, ks, padding=ks//2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        identity = self.skip(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return torch.relu(out)


class OffTargetCNN(nn.Module):
    """Deeper CNN with residual connections"""
    def __init__(self):
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv1d(4, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.res1 = ResidualBlock(64, 128)
        self.res2 = ResidualBlock(128, 256)
        self.res3 = ResidualBlock(256, 512)
        self.res4 = ResidualBlock(512, 512)
        self.res5 = ResidualBlock(512, 256)

        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.init(x)
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x = self.pool(x)
        x = self.res3(x)
        x = self.pool(x)
        x = self.res4(x)
        x = self.pool(x)
        x = self.res5(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.fc(x)
        return x


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        pt = p * labels + (1 - p) * (1 - labels)
        focal_weight = self.alpha * torch.pow(1.0 - pt, self.gamma)
        return (focal_weight * ce).mean()


def train_epoch(model, loader, opt, device, loss_fn, acc_steps=2):
    """Train one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    opt.zero_grad()

    for idx, (X, y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device).view(-1, 1)

        logits = model(X)
        loss = loss_fn(logits, y) / acc_steps
        loss.backward()

        if (idx + 1) % acc_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

        total_loss += loss.item() * acc_steps

    if len(loader) % acc_steps != 0:
        opt.step()
        opt.zero_grad()

    return total_loss / len(loader.dataset)


def validate(model, loader, device):
    """Validate"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            preds = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy().flatten())

    if len(set(all_labels)) == 1:
        return 0.5
    return roc_auc_score(all_labels, all_preds)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/raw/crisprofft/CRISPRoffT_all_targets.txt')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=30)
    args = parser.parse_args()

    # Load data
    print("\n=== Loading Data ===")
    X, y = load_crisprofft_data(args.data_path)

    # Split train/val (80/20) BEFORE weighted sampling
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"\nTrain: ON={int((y_train==0).sum())}, OFF={int((y_train==1).sum())}")
    print(f"Val:   ON={int((y_val==0).sum())}, OFF={int((y_val==1).sum())}")

    # Create weighted sampler (oversample minority class)
    class_weights = np.array([
        (y_train == 1).sum() / (y_train == 0).sum(),  # weight for ON (minority)
        1.0  # weight for OFF (majority)
    ])
    sample_weights = np.array([class_weights[int(label)] for label in y_train])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    print(f"Class weights for sampling: ON={class_weights[0]:.2f}, OFF={class_weights[1]:.2f}")

    # Datasets
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    print("\n=== Building Model ===")
    model = OffTargetCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # Training
    print(f"\n=== Training (Focal Loss + Weighted Sampling) ===")
    best_auroc = 0
    no_improve = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion, acc_steps=2)
        val_auroc = validate(model, val_loader, device)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | AUROC: {val_auroc:.4f} | LR: {lr:.7f}")

        scheduler.step()

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            no_improve = 0
            torch.save(model.state_dict(), "best_off_target_model.pt")
            print(f"  -> New best: {best_auroc:.4f}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0 and epoch > 0:
            print(f"  [Checkpoint] Best so far: {best_auroc:.4f}")

    print(f"\n=== Complete ===")
    print(f"Best AUROC: {best_auroc:.4f}")
    print(f"Model: best_off_target_model.pt")


if __name__ == "__main__":
    main()
