#!/usr/bin/env python3
"""
Off-target prediction model using CRISPRoffT dataset.
Target: AUROC >= 0.99
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def load_crisprofft_data(data_path, max_samples=None):
    """Load CRISPRoffT dataset"""
    seqs, labels = [], []
    
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
                target_status = parts[33]  # "ON" or "OFF"
                
                # Convert ON/OFF to binary: OFF=1 (off-target), ON=0 (on-target)
                if target_status not in ["ON", "OFF"]:
                    continue
                
                label = 1 if target_status == "OFF" else 0
                
                # Skip invalid sequences
                if not guide_seq or len(guide_seq) < 20:
                    continue
                
                seqs.append(guide_seq)
                labels.append(label)
                
                if max_samples and len(seqs) >= max_samples:
                    break
            except (IndexError, ValueError):
                continue
    
    print(f"Loaded {len(seqs)} samples")
    
    # One-hot encode
    X = np.array([one_hot_encode(seq) for seq in seqs]).astype(np.float32)
    y = np.array(labels).astype(np.float32)
    
    return X, y

class OffTargetCNN(nn.Module):
    """Advanced CNN for off-target prediction"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 128, kernel_size=8, padding=3)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=8, padding=3)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=8, padding=3)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    criterion = nn.BCELoss()
    
    for batch_feats, batch_labels in train_loader:
        batch_feats = batch_feats.to(device)
        batch_labels = batch_labels.to(device).view(-1, 1)
        
        optimizer.zero_grad()
        preds = model(batch_feats)
        loss = criterion(preds, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch_labels)
    
    return total_loss / len(train_loader.dataset)

def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    all_preds, all_labels = [], []
    criterion = nn.BCELoss()
    total_loss = 0
    
    with torch.no_grad():
        for batch_feats, batch_labels in val_loader:
            batch_feats = batch_feats.to(device)
            batch_labels = batch_labels.to(device)
            preds = model(batch_feats)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(batch_labels.cpu().numpy().flatten())
            loss = criterion(preds, batch_labels.view(-1, 1))
            total_loss += loss.item() * len(batch_labels)
    
    auroc = roc_auc_score(all_labels, all_preds)
    avg_loss = total_loss / len(val_loader.dataset)
    return auroc, avg_loss

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/raw/crisprofft/CRISPRoffT_all_targets.txt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    X, y = load_crisprofft_data(args.data_path)
    
    print(f"Data shape: {X.shape}, Labels: {y.shape}")
    print(f"OFF-target samples: {int(y.sum())} / {len(y)}")
    
    # Split into train/val (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    
    # Model and optimizer
    model = OffTargetCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_auroc = 0
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_auroc, val_loss = validate(model, val_loader, device)
        
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:02d} | Loss: {train_loss:.4f} | AUROC: {val_auroc:.4f} | LR: {lr:.6f}")
        
        scheduler.step(val_auroc)
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), "best_offtarget_model_v4.pt")
    
    print(f"Training finished. Best AUROC: {best_auroc:.4f}")

if __name__ == "__main__":
    main()
