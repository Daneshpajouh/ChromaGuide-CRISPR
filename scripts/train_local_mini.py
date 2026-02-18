#!/usr/bin/env python
"""Mini local training script for ChromaGuide - quick test on Mac Studio MPS"""
import os
import sys
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MiniConfig:
    batch_size = 16
    num_epochs = 5
    learning_rate = 1e-4
    seq_len = 23
    hidden_dim = 128
    num_samples = 500
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

def spearman_correlation(preds, targets):
    """Simple Spearman correlation without scipy"""
    def rankdata(x):
        temp = x.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(x))
        return ranks
    preds_rank = rankdata(preds)
    targets_rank = rankdata(targets)
    n = len(preds)
    d = preds_rank - targets_rank
    return 1 - (6 * np.sum(d**2)) / (n * (n**2 - 1))

def create_synthetic_data(config):
    logger.info(f"Creating synthetic dataset with {config.num_samples} samples...")
    sequences = torch.randint(0, 4, (config.num_samples, config.seq_len))
    one_hot = torch.zeros(config.num_samples, config.seq_len, 4)
    for i in range(config.num_samples):
        for j in range(config.seq_len):
            one_hot[i, j, sequences[i, j]] = 1
    targets = torch.rand(config.num_samples, 1)
    return one_hot, targets

class SimpleCNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    preds, targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            preds.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
    preds, targets = np.array(preds).flatten(), np.array(targets).flatten()
    spearman = spearman_correlation(preds, targets)
    return total_loss / len(dataloader), spearman

def main():
    config = MiniConfig()
    logger.info(f"=== ChromaGuide Mini Local Training ===")
    logger.info(f"Device: {config.device}")
    logger.info(f"Batch size: {config.batch_size}, Epochs: {config.num_epochs}")
    
    X, y = create_synthetic_data(config)
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.batch_size)
    
    model = SimpleCNNModel(config).to(config.device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    logger.info("Starting training...")
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
        val_loss, spearman = evaluate(model, val_loader, criterion, config.device)
        logger.info(f"Epoch {epoch+1}/{config.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Spearman: {spearman:.4f}")
    
    logger.info("=== Training Complete ===")
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/mini_model_local.pt')
    logger.info("Model saved to checkpoints/mini_model_local.pt")

if __name__ == '__main__':
    main()
