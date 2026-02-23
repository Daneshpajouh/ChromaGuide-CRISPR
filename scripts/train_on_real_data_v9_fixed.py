#!/usr/bin/env python3
"""
V9 MULTIMODAL: Maximum-power architecture to EXCEED Rho >= 0.911 target
Simplified for actual CSV data structure
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}\n")


class TransformerSequenceEncoder(nn.Module):
    """Transformer for sequence encoding."""
    def __init__(self, d_model=256, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Embedding
        self.embed = nn.Linear(4, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 30, d_model) * 0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, 120) = 30bp * 4
        x = x.view(x.size(0), 30, 4)  # (batch, 30, 4)
        x = self.embed(x)  # (batch, 30, d_model)
        x += self.pos_encoding
        x = self.transformer(x)
        x = self.ln(x)
        x = x.mean(dim=1)  # (batch, d_model)
        return x


class DeepFusion(nn.Module):
    """Deep cross-modal fusion."""
    def __init__(self, d_seq=256, d_epi=256, d_fusion=512):
        super().__init__()

        # Project epi to d_epi
        self.epi_proj = nn.Linear(690, d_epi)

        # Attention
        self.attn = nn.MultiheadAttention(d_fusion, num_heads=8, batch_first=True)

        # Projection
        self.seq_proj = nn.Linear(d_seq, d_fusion)
        self.epi_proj2 = nn.Linear(d_epi, d_fusion)

        # Fusion MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_fusion * 2, d_fusion * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_fusion * 2, d_fusion),
            nn.ReLU(),
            nn.Linear(d_fusion, 256)
        )

    def forward(self, seq_feat, epi_feat):
        epi_feat = self.epi_proj(epi_feat)

        seq_proj = self.seq_proj(seq_feat).unsqueeze(1)  # (batch, 1, d_fusion)
        epi_proj = self.epi_proj2(epi_feat).unsqueeze(1)  # (batch, 1, d_fusion)

        # Cross attention
        attn_out, _ = self.attn(seq_proj, epi_proj, epi_proj)

        # Combine
        combined = torch.cat([seq_proj.squeeze(1), attn_out.squeeze(1)], dim=1)
        fused = self.mlp(combined)

        return fused


class MultimodalV9(nn.Module):
    """V9 Multimodal Efficacy Predictor."""
    def __init__(self):
        super().__init__()
        self.seq_encoder = TransformerSequenceEncoder(d_model=256, n_layers=3)
        self.fusion = DeepFusion(d_seq=256, d_epi=256, d_fusion=512)

        # Output head (Beta regression)
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, seq, epi):
        seq_feat = self.seq_encoder(seq)  # (batch, 256)
        fused = self.fusion(seq_feat, epi)  # (batch, 256)
        params = self.head(fused)  # (batch, 2)

        alpha = torch.nn.functional.softplus(params[:, 0]) + 1e-3
        beta = torch.nn.functional.softplus(params[:, 1]) + 1e-3

        return alpha, beta


def load_csv_data(data_dir='/Users/studio/Desktop/PhD/Proposal/data/processed/split_a'):
    """Load data from CSV files."""
    print("Loading CSV data...")

    cell_types = ['HCT116', 'HEK293T', 'HeLa']
    train_dfs, val_dfs, test_dfs = [], [], []

    for ct in cell_types:
        try:
            train_dfs.append(pd.read_csv(f'{data_dir}/{ct}_train.csv'))
            val_dfs.append(pd.read_csv(f'{data_dir}/{ct}_validation.csv'))
            test_dfs.append(pd.read_csv(f'{data_dir}/{ct}_test.csv'))
            print(f"  ✓ {ct}")
        except:
            pass

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    # Encode sequences
    def encode_seqs(seqs):
        mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0.25,0.25,0.25,0.25]}
        encoded = []
        for seq in seqs:
            seq = str(seq).upper() if not pd.isna(seq) else 'A'*30
            vec = np.zeros((30, 4), dtype=np.float32)
            for i, c in enumerate(seq[:30]):
                vec[i] = mapping.get(c, mapping['N'])
            encoded.append(vec.flatten())
        return np.array(encoded)

    X_train = encode_seqs(train_df['sequence'].values)
    X_val = encode_seqs(val_df['sequence'].values)
    X_test = encode_seqs(test_df['sequence'].values)

    # Targets
    y_train = train_df['efficiency'].values.astype(np.float32)
    y_val = val_df['efficiency'].values.astype(np.float32)
    y_test = test_df['efficiency'].values.astype(np.float32)

    # Epigenomics features
    feat_cols = sorted([c for c in train_df.columns if c.startswith('feat_')],
                       key=lambda x: int(x.split('_')[1]))

    if feat_cols:
        epi_train = train_df[feat_cols].values.astype(np.float32)
        epi_val = val_df[feat_cols].values.astype(np.float32)
        epi_test = test_df[feat_cols].values.astype(np.float32)

        # Pad to 690
        if epi_train.shape[1] < 690:
            pad = 690 - epi_train.shape[1]
            epi_train = np.hstack([epi_train, np.zeros((epi_train.shape[0], pad))])
            epi_val = np.hstack([epi_val, np.zeros((epi_val.shape[0], pad))])
            epi_test = np.hstack([epi_test, np.zeros((epi_test.shape[0], pad))])
    else:
        epi_train = np.random.randn(len(y_train), 690).astype(np.float32)
        epi_val = np.random.randn(len(y_val), 690).astype(np.float32)
        epi_test = np.random.randn(len(y_test), 690).astype(np.float32)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}\n")

    return (X_train, y_train, epi_train), (X_val, y_val, epi_val), (X_test, y_test, epi_test)


def beta_loss(alpha, beta, targets):
    """Beta regression loss."""
    targets = targets * 0.95 + 0.025  # Label smoothing
    log_beta = (
        torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta) +
        (alpha - 1) * torch.log(targets + 1e-6) +
        (beta - 1) * torch.log(1 - targets + 1e-6)
    )
    return -log_beta.mean()


def train_model(model, train_data, val_data, test_data, seed=0, epochs=500):
    """Train a V9 model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train, epi_train = train_data
    X_val, y_val, epi_val = val_data
    X_test, y_test, epi_test = test_data

    # Convert
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    epi_train = torch.FloatTensor(epi_train).to(device)

    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    epi_val = torch.FloatTensor(epi_val).to(device)

    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    epi_test = torch.FloatTensor(epi_test).to(device)

    model = model.to(device)

    # DataLoader
    train_ds = TensorDataset(X_train, y_train, epi_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-5)

    best_val_rho = -1
    patience = 0
    best_state = None

    print(f"\nTraining seed {seed}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for seq_b, y_b, epi_b in train_loader:
            alpha, beta = model(seq_b, epi_b)
            loss = beta_loss(alpha, beta, y_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Val
        with torch.no_grad():
            model.eval()
            alpha_v, beta_v = model(X_val, epi_val)
            val_pred = alpha_v / (alpha_v + beta_v)
            val_rho, _ = spearmanr(y_val.cpu().numpy(), val_pred.cpu().numpy())

            alpha_t, beta_t = model(X_test, epi_test)
            test_pred = alpha_t / (alpha_t + beta_t)
            test_rho, _ = spearmanr(y_test.cpu().numpy(), test_pred.cpu().numpy())

        if epoch % 20 == 0:
            print(f"  E{epoch:3d} | Loss {train_loss:.4f} | Val {val_rho:.4f} | Test {test_rho:.4f}")

        if val_rho > best_val_rho:
            best_val_rho = val_rho
            patience = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if patience >= 30:
            break

    if best_state:
        model.load_state_dict(best_state)

    with torch.no_grad():
        model.eval()
        alpha_t, beta_t = model(X_test, epi_test)
        test_pred = alpha_t / (alpha_t + beta_t)
        final_rho, _ = spearmanr(y_test.cpu().numpy(), test_pred.cpu().numpy())

    print(f"  Final Test Rho: {final_rho:.4f}")

    return model, final_rho, test_pred


def main():
    print("="*70)
    print("V9 MULTIMODAL: EXCEED 0.911 RHO TARGET")
    print("="*70 + "\n")

    train_data, val_data, test_data = load_csv_data()

    models = []
    preds = []
    rhos = []

    for i in range(5):
        print(f"\n{'='*70}")
        print(f"MODEL {i+1}/5 (seed={i})")
        print(f"{'='*70}")

        model = MultimodalV9()
        trained_model, rho, pred = train_model(model, train_data, val_data, test_data, seed=i, epochs=500)

        models.append(trained_model)
        preds.append(pred.cpu().numpy())
        rhos.append(rho)

        # Save
        torch.save(trained_model.state_dict(), f'/Users/studio/Desktop/PhD/Proposal/models/multimodal_v9_seed{i}.pt')

    # Ensemble
    ens_pred = np.mean(preds, axis=0)
    ens_rho, _ = spearmanr(test_data[1], ens_pred)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Individual: {[f'{r:.4f}' for r in rhos]}")
    print(f"Mean: {np.mean(rhos):.4f} ± {np.std(rhos):.4f}")
    print(f"Ensemble: {ens_rho:.4f}")
    print(f"\nTarget: 0.911")
    print(f"Achievement: {ens_rho/0.911*100:.1f}%")

    if ens_rho >= 0.911:
        print("\n✅ TARGET ACHIEVED!")
    else:
        print(f"\n⚠️ Gap: {0.911-ens_rho:.4f}")

    # Save ensemble
    torch.save({
        'models': [m.state_dict() for m in models],
        'predictions': ens_pred,
        'rho_scores': rhos,
        'ensemble_rho': ens_rho
    }, '/Users/studio/Desktop/PhD/Proposal/models/multimodal_v9_ensemble.pt')

    print(f"\n✓ Ensemble saved\n")


if __name__ == '__main__':
    main()
