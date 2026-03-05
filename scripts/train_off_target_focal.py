#!/usr/bin/env python3
"""
Off-target prediction model with FOCAL LOSS for extreme class imbalance.
Target: AUROC >= 0.99

This addresses the 99.54% OFF-target imbalance with focal loss (alpha=0.25, gamma=2.0)
which downweights easy samples and focuses on hard examples.
"""

import os
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def resolve_device(device_arg: str) -> torch.device:
    """Resolve device string with graceful fallback."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def one_hot_encode(seq, length=32):
    """One-hot encode a DNA sequence."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((4, length))
    s = seq.upper()[:length]
    for j, c in enumerate(s):
        if j < length and c in mapping:
            encoded[mapping[c], j] = 1
    return encoded


def encode_pair(guide_seq, target_seq, length=32):
    """Encode guide-target pair as channels: guide(4) + target(4) + mismatch(1)."""
    g = one_hot_encode(guide_seq, length=length)
    t = one_hot_encode(target_seq, length=length)
    g_idx = np.argmax(g, axis=0)
    t_idx = np.argmax(t, axis=0)
    g_valid = g.sum(axis=0) > 0
    t_valid = t.sum(axis=0) > 0
    mismatch = ((g_idx != t_idx) & g_valid & t_valid).astype(np.float32)[None, :]
    return np.concatenate([g, t, mismatch], axis=0), float(mismatch.sum())

def load_crisprofft_data(data_path, max_samples=None):
    """Load CRISPRoffT dataset with validation"""
    pair_feats, aux_feats, labels = [], [], []

    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:  # Skip header
                continue

            parts = line.strip().split('\t')
            if len(parts) < 35:  # Need at least 35 columns
                continue

            try:
                # Column 22 (index 21) = Guide_sequence
                # Column 34 (index 33) = Identity (ON/OFF)
                guide_seq = parts[21]
                target_seq = parts[22]
                target_status = parts[33]  # "ON" or "OFF"

                # Convert ON/OFF to binary: OFF=1 (off-target), ON=0 (on-target)
                if target_status not in ["ON", "OFF"]:
                    continue

                label = 1 if target_status == "OFF" else 0

                # Skip invalid sequences
                if not guide_seq or not target_seq or len(guide_seq) < 20 or len(target_seq) < 20:
                    continue

                # Auxiliary scalar features from CRISPRoffT columns.
                mismatch_raw = parts[34]
                bulge_raw = parts[35]
                try:
                    mismatch_count = float(mismatch_raw)
                except Exception:
                    mismatch_count = np.nan
                if bulge_raw == "NULL":
                    bulge_count = 0.0
                else:
                    try:
                        bulge_count = float(bulge_raw)
                    except Exception:
                        bulge_count = 0.0

                pair_ch, inferred_mismatch = encode_pair(guide_seq, target_seq, length=32)
                if np.isnan(mismatch_count):
                    mismatch_count = inferred_mismatch

                pair_feats.append(pair_ch)
                # Normalize mismatch count to roughly [0,1] scale.
                aux_feats.append([mismatch_count / 16.0, bulge_count / 4.0])
                labels.append(label)

                if max_samples and len(pair_feats) >= max_samples:
                    break
            except (IndexError, ValueError):
                continue

    print(f"Loaded {len(pair_feats)} raw guide-target pairs")

    # One-hot encode pair channels + scalar aux.
    X = np.array(pair_feats).astype(np.float32)
    aux = np.array(aux_feats).astype(np.float32)
    y = np.array(labels).astype(np.float32)

    # VALIDATION: Check encoding quality
    print(f"Encoded shape: {X.shape}")
    print(f"Data type: {X.dtype}")
    print(f"Non-zero elements in batch: {np.sum(X[:10] > 0)}")
    print(f"Sample encoding (should be 9 x 32): {X[0].shape}")
    if X[0].sum() == 0:
        raise ValueError("ERROR: First sample encoding is all zeros!")

    # Check label distribution
    print(f"Label distribution - ON: {(y==0).sum()}, OFF: {(y==1).sum()}")

    return X, aux, y


class OffTargetCNN(nn.Module):
    """Advanced CNN for off-target prediction with tunable capacity."""
    def __init__(
        self,
        base_channels: int = 256,
        fc_hidden: int = 256,
        conv_dropout: float = 0.4,
        fc_dropout: float = 0.3,
    ):
        super().__init__()
        mid_channels = max(64, base_channels // 2)
        high_channels = max(base_channels, base_channels * 2)
        # Pair channels: guide(4) + target(4) + mismatch(1) = 9 channels
        self.conv1 = nn.Conv1d(9, base_channels, kernel_size=8, padding=3)
        self.conv2 = nn.Conv1d(base_channels, high_channels, kernel_size=8, padding=3)
        self.conv3 = nn.Conv1d(high_channels, base_channels, kernel_size=8, padding=3)
        self.conv4 = nn.Conv1d(base_channels, mid_channels, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(conv_dropout)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.bn2 = nn.BatchNorm1d(high_channels)
        self.bn3 = nn.BatchNorm1d(base_channels)
        self.bn4 = nn.BatchNorm1d(mid_channels)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(mid_channels + 2, fc_hidden),
            nn.ReLU(),
            nn.Dropout(conv_dropout),
            nn.Linear(fc_hidden, max(64, fc_hidden // 2)),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(max(64, fc_hidden // 2), 1)
        )

    def forward(self, x, aux):
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.bn2(torch.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.bn3(torch.relu(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.bn4(torch.relu(self.conv4(x)))
        x = self.pool(x)
        x = self.gap(x).squeeze(-1)
        x = torch.cat([x, aux], dim=1)
        x = self.fc(x)
        return x


class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance.

    Addresses class imbalance by down-weighting easy (correct) samples
    and focusing learning on hard (misclassified) samples.

    Formula: FL = -alpha * (1-pt)^gamma * log(pt)
    where:
    - pt: model predicted probability for true class
    - alpha: balancing factor (0.25 in our case)
    - gamma: focusing parameter (2.0 in our case)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model output (batch_size, 1)
            targets: Binary labels (batch_size, 1)
        Returns:
            Scalar focal loss
        """
        # Convert logits to probabilities
        p = torch.sigmoid(logits)

        # Calculate focal loss
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # pt is the probability of target class
        pt = p * targets + (1 - p) * (1 - targets)

        # Apply focal weighting
        focal_weight = self.alpha * torch.pow(1.0 - pt, self.gamma)
        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


def train_epoch(model, train_loader, optimizer, device, criterion):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch_feats, batch_aux, batch_labels in train_loader:
        batch_feats = batch_feats.to(device)
        batch_aux = batch_aux.to(device)
        batch_labels = batch_labels.to(device).view(-1, 1)

        optimizer.zero_grad()
        logits = model(batch_feats, batch_aux)
        loss = criterion(logits, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(batch_labels)

    return total_loss / len(train_loader.dataset)


def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch_feats, batch_aux, batch_labels in val_loader:
            batch_feats = batch_feats.to(device)
            batch_aux = batch_aux.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_feats, batch_aux)
            # Convert logits to probabilities for AUROC calculation
            preds_prob = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(preds_prob)
            all_labels.extend(batch_labels.cpu().numpy().flatten())

    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    return auroc, auprc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/raw/crisprofft/CRISPRoffT_all_targets.txt')
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--base_channels', type=int, default=256)
    parser.add_argument('--fc_hidden', type=int, default=256)
    parser.add_argument('--conv_dropout', type=float, default=0.4)
    parser.add_argument('--fc_dropout', type=float, default=0.3)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--scheduler_factor', type=float, default=0.5)
    parser.add_argument('--scheduler_patience', type=int, default=5)
    parser.add_argument('--early_stop_patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--output_json', type=str, default='')
    parser.add_argument('--model_out', type=str, default='')
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from {args.data_path}...")
    max_samples = args.max_samples if args.max_samples > 0 else None
    X, aux, y = load_crisprofft_data(args.data_path, max_samples=max_samples)

    print(f"Data shape: {X.shape}, Labels: {y.shape}")
    print(f"OFF-target samples: {int(y.sum())} / {len(y)}")

    # Stratified split into train/val (80/20)
    X_train, X_val, aux_train, aux_val, y_train, y_val = train_test_split(
        X, aux, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(aux_train),
        torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(aux_val),
        torch.from_numpy(y_val),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model and optimizer
    model = OffTargetCNN(
        base_channels=args.base_channels,
        fc_hidden=args.fc_hidden,
        conv_dropout=args.conv_dropout,
        fc_dropout=args.fc_dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    # Use Focal Loss for extreme class imbalance
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    best_auroc = 0
    best_auprc = 0
    best_epoch = 0
    patience_counter = 0

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        val_auroc, val_auprc = validate(model, val_loader, device)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | "
            f"AUROC: {val_auroc:.4f} | AUPRC: {val_auprc:.4f} | LR: {lr:.6f}"
        )

        scheduler.step(val_auroc)
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_auprc = val_auprc
            best_epoch = epoch + 1
            patience_counter = 0
            model_out = args.model_out.strip() or "best_offtarget_model_focal.pt"
            model_dir = os.path.dirname(model_out)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), model_out)
        else:
            patience_counter += 1
            # Early stopping if no improvement for configured patience
            if patience_counter >= args.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch+1}: "
                    f"no improvement for {args.early_stop_patience} epochs"
                )
                break

    out_json = args.output_json.strip()
    if not out_json:
        out_json = f"results/runs/offtarget_focal_seed{args.seed}.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "data_path": args.data_path,
                "max_samples": args.max_samples,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "base_channels": args.base_channels,
                "fc_hidden": args.fc_hidden,
                "conv_dropout": args.conv_dropout,
                "fc_dropout": args.fc_dropout,
                "focal_alpha": args.focal_alpha,
                "focal_gamma": args.focal_gamma,
                "scheduler_factor": args.scheduler_factor,
                "scheduler_patience": args.scheduler_patience,
                "early_stop_patience": args.early_stop_patience,
                "best_epoch": best_epoch,
                "best_auroc": float(best_auroc),
                "best_auprc": float(best_auprc),
                "model_out": args.model_out.strip() or "best_offtarget_model_focal.pt",
                "n_samples": int(len(y)),
                "n_off": int(y.sum()),
                "n_on": int((y == 0).sum()),
            },
            f,
            indent=2,
        )

    print(f"Training finished. Best AUROC: {best_auroc:.4f} | Best AUPRC: {best_auprc:.4f}")
    print(f"Model saved to {args.model_out.strip() or 'best_offtarget_model_focal.pt'}")
    print(f"Metrics saved to {out_json}")


if __name__ == "__main__":
    main()
