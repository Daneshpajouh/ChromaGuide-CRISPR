#!/usr/bin/env python
"""ChromaGuide local training with actual model architecture on Mac Studio MPS"""
import os
import sys
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalConfig:
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    seq_len = 23
    hidden_dim = 256
    num_layers = 2
    num_samples = 1000
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

def spearman_correlation(preds, targets):
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
    # Synthetic sgRNA sequences (one-hot: A=0, C=1, G=2, T=3)
    sequences = torch.randint(0, 4, (config.num_samples, config.seq_len))
    one_hot = torch.zeros(config.num_samples, config.seq_len, 4)
    for i in range(config.num_samples):
        for j in range(config.seq_len):
            one_hot[i, j, sequences[i, j]] = 1
    # Create correlated targets based on GC content and position bias
    gc_content = (sequences == 1).float().sum(1) + (sequences == 2).float().sum(1)
    gc_content = gc_content / config.seq_len
    pos_bias = torch.sigmoid(sequences[:, 10:15].float().mean(1) - 1.5)
    targets = 0.5 * gc_content + 0.3 * pos_bias + 0.2 * torch.rand(config.num_samples)
    targets = targets.unsqueeze(1)
    return one_hot, targets

class DNABERTLikeEncoder(nn.Module):
    """Simplified DNABERT-like encoder for local testing"""
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Linear(4, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim, nhead=4, dim_feedforward=config.hidden_dim*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x.mean(dim=1)  # Global average pooling

class MambaLikeBlock(nn.Module):
    """Simplified Mamba-like SSM block for local testing"""
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        # Conv processing
        x_conv = x.permute(0, 2, 1)
        x_conv = self.conv(x_conv).permute(0, 2, 1)
        # Gate
        gate = torch.sigmoid(self.gate(x))
        x = self.proj(x_conv) * gate
        return x + residual

class ChromaGuideMiniModel(nn.Module):
    """Mini ChromaGuide model with DNABERT + Mamba-like architecture"""
    def __init__(self, config):
        super().__init__()
        self.dnabert = DNABERTLikeEncoder(config)
        self.mamba_blocks = nn.ModuleList([MambaLikeBlock(config.hidden_dim) for _ in range(2)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # DNABERT branch
        dnabert_out = self.dnabert(x)
        
        # Mamba branch (process sequence)
        mamba_in = x.permute(0, 2, 1)  # (B, 4, L) -> for conv
        mamba_in = nn.functional.linear(x, torch.randn(256, 4, device=x.device))  # Simple projection
        for block in self.mamba_blocks:
            mamba_in = block(mamba_in)
        mamba_out = mamba_in.mean(dim=1)
        
        # Fusion
        fused = torch.cat([dnabert_out, mamba_out], dim=-1)
        x = self.dropout(torch.relu(self.fc1(fused)))
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    config = LocalConfig()
    logger.info(f"=== ChromaGuide Local Training (DNABERT + Mamba Architecture) ===")
    logger.info(f"Device: {config.device}")
    logger.info(f"Batch size: {config.batch_size}, Epochs: {config.num_epochs}")
    
    X, y = create_synthetic_data(config)
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.batch_size)
    
    model = ChromaGuideMiniModel(config).to(config.device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    criterion = nn.MSELoss()
    
    best_spearman = -1
    logger.info("Starting training...")
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
        val_loss, spearman = evaluate(model, val_loader, criterion, config.device)
        scheduler.step()
        
        if spearman > best_spearman:
            best_spearman = spearman
            torch.save(model.state_dict(), 'checkpoints/chromaguide_local_best.pt')
        
        logger.info(f"Epoch {epoch+1:2d}/{config.num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Spearman: {spearman:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    logger.info(f"=== Training Complete === Best Spearman: {best_spearman:.4f}")
    torch.save(model.state_dict(), 'checkpoints/chromaguide_local_final.pt')
    logger.info("Models saved to checkpoints/")

if __name__ == '__main__':
    main()
