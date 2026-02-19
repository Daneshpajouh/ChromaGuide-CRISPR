import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from scipy.stats import spearmanr
from pathlib import Path
import json
import logging
import sys
import argparse

# Add src to path for ChromaGuide imports
sys.path.insert(0, '/home/amird/chromaguide_experiments/src')

try:
    from chromaguide.chromaguide_model import ChromaGuideModel
    USE_CHROMAGUIDE = True
    print("Successfully imported ChromaGuideModel")
except ImportError as e:
    import traceback
    print(f"ChromaGuideModel import failed: {e}")
    traceback.print_exc()
    USE_CHROMAGUIDE = False

# Disable Torch compilation if it causes issues on cluster
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Setup Paths
MAMBA_ENV = "/home/amird/env_chromaguide"
LOCAL_CACHE = "/home/amird/.cache/huggingface/hub"
os.environ["HF_HOME"] = LOCAL_CACHE

class BetaRegressionHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = self.fc(x)
        alpha = self.softplus(out[:, 0:1]) + 1.01
        beta = self.softplus(out[:, 1:2]) + 1.01
        return alpha, beta

def beta_nll_loss(alpha, beta, target):
    dist = torch.distributions.Beta(alpha, beta)
    target = torch.clamp(target, 0.001, 0.999)
    return -dist.log_prob(target).mean()

def main():
    parser = argparse.ArgumentParser(description='ChromaGuide Production Training')
    parser.add_argument('--backbone', type=str, default='cnn_gru', choices=['cnn_gru', 'dnabert2'], help='Backbone types')
    parser.add_argument('--fusion', type=str, default='gate', choices=['gate', 'concat'], help='Fusion type')
    parser.add_argument('--use_epi', action='store_true', help='Use epigenomics')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: Backbone={args.backbone}, Fusion={args.fusion}, UseEpi={args.use_epi}")

    # Load data
    data_path = "/home/amird/chromaguide_experiments/data/real/merged.csv"
    gold_path = "/home/amird/chromaguide_experiments/test_set_GOLD.csv"

    all_df = pd.read_csv(data_path)
    test_df = pd.read_csv(gold_path)

    # Leakage Protection: Remove GOLD set samples
    gold_sequences = set(test_df['sequence'].tolist())
    train_val_df = all_df[~all_df['sequence'].isin(gold_sequences)].copy()
    
    print(f"Clean Train/Val samples: {len(train_val_df)}")

    # Split train/val
    train_df = train_val_df.sample(frac=0.9, random_state=42)
    val_df = train_val_df.drop(train_df.index)

    # Initialize Model
    if USE_CHROMAGUIDE:
        model = ChromaGuideModel(
            encoder_type=args.backbone,
            d_model=256 if args.backbone == 'cnn_gru' else 768,
            seq_len=23,
            num_epi_tracks=4,
            num_epi_bins=100,
            use_epigenomics=args.use_epi,
            use_gate_fusion=(args.fusion == 'gate'),
            dropout=0.1,
        ).to(device)
        
        # Optimization
        if args.backbone == 'dnabert2':
            # Low LR for transformer backbone
            backbone_params = [p for n, p in model.named_parameters() if 'seq_encoder' in n]
            head_params = [p for n, p in model.named_parameters() if 'seq_encoder' not in n]
            optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': 1e-5},
                {'params': head_params, 'lr': 1e-4}
            ])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            
        print(f"Model initialized: {args.backbone} with {args.fusion} fusion")
    else:
        print("CRITICAL: ChromaGuideModel import failed.")
        sys.exit(1)

    # Training Loop
    epochs = 15
    batch_size = 256
    best_val_rho = -1
    patience = 5
    no_improve = 0
    nt_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        curr_train = train_df.sample(frac=1).reset_index(drop=True)

        for i in range(0, len(curr_train), batch_size):
            batch = curr_train.iloc[i : i+batch_size]
            seqs = batch['sequence'].tolist()
            labels = torch.tensor(batch['efficiency'].values, dtype=torch.float32).to(device).unsqueeze(1)
            
            seq_tensor = torch.zeros(len(seqs), 4, 23, device=device)
            for j, seq in enumerate(seqs):
                for k, nt in enumerate(seq[:23]):
                    if nt in nt_map: seq_tensor[j, nt_map[nt], k] = 1

            output = model(seq_tensor)
            loss_dict = model.compute_loss(output, labels)
            loss = loss_dict['total_loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for i in range(0, len(val_df), batch_size):
                batch = val_df.iloc[i : i+batch_size]
                seq_tensor = torch.zeros(len(batch), 4, 23, device=device)
                for j, seq in enumerate(batch['sequence']):
                    for k, nt in enumerate(seq[:23]):
                        if nt in nt_map: seq_tensor[j, nt_map[nt], k] = 1

                output = model(seq_tensor)
                val_preds.extend(output['mu'].cpu().numpy().flatten())
                val_labels.extend(batch['efficiency'].values)

        val_rho, _ = spearmanr(val_preds, val_labels)
        print(f"Epoch {epoch+1} | Loss: {total_loss/(len(train_df)/batch_size):.4f} | Val Rho: {val_rho:.4f}")

        if val_rho > best_val_rho:
            best_val_rho = val_rho
            save_name = f"checkpoint_{args.backbone}_{args.fusion}.pt"
            torch.save(model.state_dict(), save_name)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience: break

    # Final Test
    print("Evaluating on GOLD set...")
    model.load_state_dict(torch.load(f"checkpoint_{args.backbone}_{args.fusion}.pt"))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for i in range(0, len(test_df), batch_size):
            batch = test_df.iloc[i : i+batch_size]
            seq_tensor = torch.zeros(len(batch), 4, 23, device=device)
            for j, seq in enumerate(batch['sequence']):
                for k, nt in enumerate(seq[:23]):
                    if nt in nt_map: seq_tensor[j, nt_map[nt], k] = 1
            output = model(seq_tensor)
            test_preds.extend(output['mu'].cpu().numpy().flatten())
            test_labels.extend(batch['efficiency'].values)
    
    test_rho, _ = spearmanr(test_preds, test_labels)
    print(f"FINAL GOLD Rho: {test_rho:.4f}")
    
    pd.DataFrame({'sequence': test_df['sequence'], 'efficiency': test_labels, 'prediction': test_preds}).to_csv(f"gold_results_{args.backbone}_{args.fusion}.csv", index=False)

if __name__ == '__main__':
    main()
