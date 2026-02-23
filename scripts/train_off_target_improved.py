#!/usr/bin/env python3
"""
Improved Off-target prediction with multiple enhancements for class imbalance.
Target: AUROC >= 0.99

Key improvements:
1. Deeper/wider CNN with residual connections
2. SMOTE-based oversampling + weighted sampling
3. Cosine annealing LR scheduler with warmup
4. Higher patience (30 epochs), max epochs (500)
5. Data augmentation for minority class
6. Gradient accumulation for larger effective batch size
7. d_model = 64 as per thesis proposal
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Check device availability (CUDA, MPS, or CPU)
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
    encoded = np.zeros((4, length))
    s = seq.upper()[:length]
    for j, c in enumerate(s):
        if j < length and c in mapping:
            encoded[mapping[c], j] = 1
    return encoded

def augment_sequence(encoded_seq, augment_prob=0.3):
    """
    Data augmentation for minority class:
    - Random position shifts (circular rotation)
    - Random mutation (change 1-2 positions)
    """
    seq = encoded_seq.copy()

    # Circular shift (keep same sequence, different frame)
    if np.random.random() < augment_prob:
        shift = np.random.randint(1, 3)  # Shift by 1-2 positions
        seq = np.roll(seq, shift, axis=1)

    # Small random mutations (flip a position)
    if np.random.random() < augment_prob:
        num_mutations = np.random.randint(0, 2)  # 0-1 mutations
        for _ in range(num_mutations):
            pos = np.random.randint(0, 23)
            # Randomly change to different nucleotide
            new_nuc = np.random.randint(0, 4)
            seq[:, pos] = 0
            seq[new_nuc, pos] = 1

    return seq

def load_crisprofft_data(data_path, max_samples=None):
    """Load CRISPRoffT dataset with validation"""
    seqs, labels = [], []

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

    print(f"Loaded {len(seqs)} raw sequences")

    # One-hot encode
    X = np.array([one_hot_encode(seq) for seq in seqs]).astype(np.float32)
    y = np.array(labels).astype(np.float32)

    print(f"Encoded shape: {X.shape}")
    print(f"Label distribution - ON: {(y==0).sum()}, OFF: {(y==1).sum()}")
    print(f"Class imbalance ratio: {(y==1).sum() / (y==0).sum():.2f} OFF per ON")

    return X, y


class ResidualBlock(nn.Module):
    """Residual block with batch norm and dropout"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.3)

        # Skip connection
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.skip(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = torch.relu(out)
        return out


class OffTargetCNN_Improved(nn.Module):
    """Improved CNN with residual connections and deeper architecture"""
    def __init__(self):
        super().__init__()
        # Initial convolution (4 channels -> 64)
        self.conv_init = nn.Conv1d(4, 64, kernel_size=7, padding=3)
        self.bn_init = nn.BatchNorm1d(64)

        # Residual blocks with increasing capacity
        self.res1 = ResidualBlock(64, 128, kernel_size=5)
        self.pool1 = nn.MaxPool1d(2)

        self.res2 = ResidualBlock(128, 256, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2)

        self.res3 = ResidualBlock(256, 512, kernel_size=5)
        self.pool3 = nn.MaxPool1d(2)

        self.res4 = ResidualBlock(512, 512, kernel_size=3)
        self.pool4 = nn.MaxPool1d(2)

        self.res5 = ResidualBlock(512, 256, kernel_size=3)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers with dropout
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
        # Initial conv
        x = torch.relu(self.bn_init(self.conv_init(x)))

        # Residual blocks with pooling
        x = self.res1(x)
        x = self.pool1(x)

        x = self.res2(x)
        x = self.pool2(x)

        x = self.res3(x)
        x = self.pool3(x)

        x = self.res4(x)
        x = self.pool4(x)

        x = self.res5(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.fc(x)
        return x


class FocalLoss(nn.Module):
    """Focal Loss with class weighting"""
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model output (batch_size, 1)
            targets: Binary labels (batch_size, 1)
        """
        p = torch.sigmoid(logits)

        # Base BCE loss
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )

        # pt: probability of true class
        pt = p * targets + (1 - p) * (1 - targets)

        # Focal weighting
        focal_weight = self.alpha * torch.pow(1.0 - pt, self.gamma)
        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


def train_epoch(model, train_loader, optimizer, device, criterion, accumulation_steps=4):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for batch_idx, (batch_feats, batch_labels) in enumerate(train_loader):
        batch_feats = batch_feats.to(device)
        batch_labels = batch_labels.to(device).view(-1, 1)

        # Forward pass
        logits = model(batch_feats)
        loss = criterion(logits, batch_labels)

        # Backward pass with accumulation
        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps * len(batch_labels)

    # Final update if needed
    if len(train_loader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(train_loader.dataset)


def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_feats, batch_labels in val_loader:
            batch_feats = batch_feats.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_feats)
            preds_prob = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(preds_prob)
            all_labels.extend(batch_labels.cpu().numpy().flatten())

    if len(set(all_labels)) == 1:  # All one class in validation
        return 0.5

    auroc = roc_auc_score(all_labels, all_preds)
    return auroc


def apply_smote_and_augmentation(X, y, augment_minority_only=True):
    """
    Apply SMOTE for oversampling + data augmentation
    """
    # Reshape for SMOTE: (N, 4*23) -> apply SMOTE -> reshape back
    X_flat = X.reshape(X.shape[0], -1)

    # Apply SMOTE with k_neighbors smaller for rare class
    n_samples = len(X)
    n_minority = (y == 1).sum()

    # Calculate sampling strategy: oversample minority to 50% of total
    sampling_strategy = min(0.5, (n_minority / (n_samples - n_minority)))

    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=min(5, n_minority-1), random_state=42)
    X_flat_resampled, y_resampled = smote.fit_resample(X_flat, y)
    X_resampled = X_flat_resampled.reshape(-1, 4, 23).astype(np.float32)

    print(f"After SMOTE: {len(y_resampled)} samples (ON: {(y_resampled==0).sum()}, OFF: {(y_resampled==1).sum()})")

    # Apply augmentation to minority (OFF-target) class
    if augment_minority_only:
        X_augmented = []
        for i, (x, label) in enumerate(zip(X_resampled, y_resampled)):
            X_augmented.append(x)
            # Augment minority class with 50% probability
            if label == 1 and np.random.random() < 0.5:
                X_augmented.append(augment_sequence(x))
        X_resampled = np.array(X_augmented)
        y_resampled = np.concatenate([y_resampled, y_resampled[(y_resampled == 1).nonzero()[0][:len(X_augmented)-len(y_resampled)]]])

    print(f"After augmentation: {len(y_resampled)} samples")

    return X_resampled, y_resampled


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
    print(f"\n=== Loading Data ===")
    X, y = load_crisprofft_data(args.data_path)

    # Split into train/val (80/20) BEFORE SMOTE
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"\nTrain set - ON: {(y_train==0).sum()}, OFF: {(y_train==1).sum()}")
    print(f"Val set - ON: {(y_val==0).sum()}, OFF: {(y_val==1).sum()}")

    # Apply SMOTE + augmentation to training data
    print(f"\n=== Applying SMOTE Oversampling + Augmentation ===")
    X_train_resampled, y_train_resampled = apply_smote_and_augmentation(X_train, y_train)

    # Create datasets
    train_ds = TensorDataset(torch.from_numpy(X_train_resampled), torch.from_numpy(y_train_resampled))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\nTrain batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    print(f"\n=== Building Model ===")
    model = OffTargetCNN_Improved().to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Optimizer with cosine annealing
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine annealing with warm restarts (period=30, multiplier=2)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    # Focal loss with pos_weight for class imbalance
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)

    print(f"Pos weight (for BCE): {pos_weight.item():.2f}")
    print(f"Learning rate schedule: CosineAnnealingWarmRestarts (T_0=30, multiplier=2)")
    print(f"Initial LR: {args.lr}, Min LR: 1e-6")
    print(f"Patience: {args.patience} epochs, Max epochs: {args.epochs}")

    # Training
    print(f"\n=== Training ===")
    best_auroc = 0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion, accumulation_steps=2)
        val_auroc = validate(model, val_loader, device)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | AUROC: {val_auroc:.4f} | LR: {lr:.6f}")

        scheduler.step()

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), "best_off_target_improved.pt")
            print(f"  -> New best AUROC: {best_auroc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}: no improvement for {args.patience} epochs")
                break

        # Print summary every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  [Epoch {epoch+1}] Best so far: {best_auroc:.4f} at epoch {best_epoch}")

    print(f"\n=== Training Complete ===")
    print(f"Best AUROC: {best_auroc:.4f} at epoch {best_epoch}")
    print(f"Model saved to best_off_target_improved.pt")


if __name__ == "__main__":
    main()
