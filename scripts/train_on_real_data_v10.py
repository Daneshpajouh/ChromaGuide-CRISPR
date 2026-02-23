#!/usr/bin/env python3
"""
V10 MULTIMODAL (ON-TARGET): DNABERT-2 + DeepFusion + Epigenetic Gating

Combines verified architectures from:
- DNABERT-2 (zhihan1996/DNABERT-2-117M) for sequence encoding
- DeepFusion with cross-attention for epigenomics integration
- CRISPR_DNABERT epigenetic gating mechanism for feature control

Target: Spearman Rho >= 0.911 on on-target efficacy prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
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


class EpigenoticGatingModule(nn.Module):
    """
    Epigenetic Feature Gating Module from CRISPR_DNABERT
    feature_dim -> 256 -> 512 -> 1024 -> 512 -> 256 (ReLU + Dropout(0.1))
    sigmoid gate controls epigenetic contribution
    """
    def __init__(self, feature_dim, hidden_dim=256, dropout=0.1):
        super().__init__()

        self.epi_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, seq_features, epi_features):
        """
        seq_features: (batch, hidden_dim)
        epi_features: (batch, feature_dim)
        Returns: gated_features (batch, hidden_dim)
        """
        epi_encoded = self.epi_encoder(epi_features)
        combined = torch.cat([seq_features, epi_encoded], dim=1)
        gate = self.gate(combined)
        gated = seq_features * (1 - gate) + epi_encoded * gate
        return gated


class DeepFusionModule(nn.Module):
    """
    Deep Fusion with Cross-Attention
    Projects epigenomics to hidden_dim and uses cross-attention
    """
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()

        # Epigenomics encoder: 690 -> hidden_dim
        self.epi_encoder = nn.Sequential(
            nn.Linear(690, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Cross-attention: sequence attends to epigenomics
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, seq_features, epi_features):
        """
        seq_features: (batch, hidden_dim)
        epi_features: (batch, 690)
        Returns: fused_features (batch, hidden_dim)
        """
        # Expand for attention
        seq_expanded = seq_features.unsqueeze(1)  # (batch, 1, hidden_dim)

        # Encode epigenomics
        epi_encoded = self.epi_encoder(epi_features)  # (batch, hidden_dim)
        epi_expanded = epi_encoded.unsqueeze(1)  # (batch, 1, hidden_dim)

        # Cross-attention: sequence attends to epigenomics
        attn_out, _ = self.cross_attn(
            query=seq_expanded,
            key=epi_expanded,
            value=epi_expanded
        )  # (batch, 1, hidden_dim)

        attn_out = attn_out.squeeze(1)  # (batch, hidden_dim)

        # Fusion
        combined = torch.cat([seq_features, attn_out], dim=1)
        fused = self.fusion(combined)

        return fused


class DNABERTMultimodalV10(nn.Module):
    """
    V10 Multimodal On-target Efficacy Predictor

    Architecture:
    1. DNABERT-2 for sequence encoding (BPE tokenization)
    2. DeepFusion for epigenomics-sequence integration
    3. Epigenetic gating for feature control
    4. Beta regression head for efficacy prediction
    """
    def __init__(self, dnabert_model_name="zhihan1996/DNABERT-2-117M", hidden_dim=256, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 1. DNABERT-2 encoder
        self.tokenizer = AutoTokenizer.from_pretrained(dnabert_model_name)
        self.dnabert = AutoModel.from_pretrained(dnabert_model_name)
        self.dnabert_dim = self.dnabert.config.hidden_size  # 768

        # Unfreeze last 6 layers for fine-tuning
        for param in self.dnabert.parameters():
            param.requires_grad = False
        for param in self.dnabert.encoder.layer[-6:].parameters():
            param.requires_grad = True

        # Project to hidden_dim
        self.seq_proj = nn.Sequential(
            nn.Linear(self.dnabert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. DeepFusion module
        self.fusion = DeepFusionModule(hidden_dim, num_heads=8, dropout=dropout)

        # 3. Epigenetic gating (feature control)
        self.epi_gating = EpigenoticGatingModule(690, hidden_dim, dropout)

        # 4. Beta regression head
        # Output: alpha and beta parameters for Beta distribution
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # alpha, beta
            nn.Softplus()  # ensure positive
        )

    def forward(self, sequences, epi_features):
        """
        sequences: list of DNA sequences (30bp)
        epi_features: (batch, 690) epigenomic features
        """
        # Tokenize and encode with DNABERT-2
        tokens = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        dnabert_out = self.dnabert(**tokens).last_hidden_state  # (batch, seq_len, 768)

        # Project sequence to hidden_dim
        seq_repr = self.seq_proj(dnabert_out[:, 0, :])  # [CLS] token

        # DeepFusion: integrate epigenomics
        fused = self.fusion(seq_repr, epi_features)

        # Epigenetic gating: control feature contribution
        gated = self.epi_gating(fused, epi_features)

        # Beta regression
        params = self.beta_head(gated)  # (batch, 2)

        return params  # alpha, beta


def load_multimodal_data(split='split_a'):
    """Load multimodal data from CSV"""
    print(f"Loading multimodal data (split: {split})...")

    data_dir = Path('/Users/studio/Desktop/PhD/Proposal/data/processed') / split

    sequences = []
    efficiencies = []
    epigenomics = []

    for cell_type in ['HCT116', 'HEK293T', 'HeLa']:
        csv_file = data_dir / f'{cell_type}_train.csv'
        if not csv_file.exists():
            continue

        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            seq = str(row['sequence']).upper()
            if len(seq) < 30:
                continue

            efficiency = float(row['efficiency'])
            if not (0 <= efficiency <= 1):
                continue

            # Extract epigenomic features (feat_0, feat_1, ..., feat_689)
            epi_feats = []
            for i in range(690):
                col = f'feat_{i}'
                if col in df.columns:
                    epi_feats.append(float(row[col]))

            if len(epi_feats) == 690:
                sequences.append(seq[:30])
                efficiencies.append(efficiency)
                epigenomics.append(np.array(epi_feats, dtype=np.float32))

    print(f"  Loaded {len(sequences)} samples")

    y = np.array(efficiencies, dtype=np.float32)
    X_epis = np.array(epigenomics, dtype=np.float32)

    print(f"  Efficiency range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Epigenomics shape: {X_epis.shape}\n")

    # Split into train/val/test
    np.random.seed(42)
    n = len(sequences)
    idx = np.random.permutation(n)

    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    train_seqs = [sequences[i] for i in train_idx]
    val_seqs = [sequences[i] for i in val_idx]
    test_seqs = [sequences[i] for i in test_idx]

    train_epis = X_epis[train_idx]
    val_epis = X_epis[val_idx]
    test_epis = X_epis[test_idx]

    train_y = y[train_idx]
    val_y = y[val_idx]
    test_y = y[test_idx]

    print(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}\n")

    return (train_seqs, train_epis, train_y), (val_seqs, val_epis, val_y), (test_seqs, test_epis, test_y)


def beta_regression_loss(alpha, beta, y_true, label_smooth=0.95):
    """Beta regression loss with label smoothing"""
    y_smooth = y_true * label_smooth + (1 - label_smooth) * 0.5
    y_smooth = torch.clamp(y_smooth, 1e-6, 1 - 1e-6)

    # Log Beta likelihood
    log_beta_fn = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    loss = -log_beta_fn + (alpha - 1) * torch.log(y_smooth) + (beta - 1) * torch.log(1 - y_smooth)

    return loss.mean()


def train_model(model, train_data, val_data, epochs=100, seed=0):
    """Train model with beta regression"""

    model = model.to(device)

    train_seqs, train_epis, train_y = train_data
    val_seqs, val_epis, val_y = val_data

    optimizer = optim.AdamW([
        {'params': model.dnabert.parameters(), 'lr': 2e-5},
        {'params': model.seq_proj.parameters(), 'lr': 5e-4},
        {'params': model.fusion.parameters(), 'lr': 5e-4},
        {'params': model.epi_gating.parameters(), 'lr': 5e-4},
        {'params': model.beta_head.parameters(), 'lr': 5e-4}
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2, eta_min=1e-6)

    best_rho = -1
    patience = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0

        batch_size = 50
        for i in range(0, len(train_seqs), batch_size):
            batch_seqs = train_seqs[i:i+batch_size]
            batch_epis = torch.FloatTensor(train_epis[i:i+batch_size]).to(device)
            batch_y = torch.FloatTensor(train_y[i:i+batch_size]).to(device)

            params = model(batch_seqs, batch_epis)
            alpha, beta = params[:, 0], params[:, 1]

            loss = beta_regression_loss(alpha, beta, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_epis_t = torch.FloatTensor(val_epis).to(device)
            params = model(val_seqs, val_epis_t)

            # Use alpha/(alpha+beta) as point estimate
            alpha, beta = params[:, 0], params[:, 1]
            pred = (alpha / (alpha + beta)).cpu().numpy()

            rho, pval = spearmanr(val_y, pred)

        if epoch % 20 == 0:
            print(f"  E{epoch:3d} | Loss {train_loss:.4f} | Val Rho {rho:.4f}")

        if rho > best_rho:
            best_rho = rho
            patience = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if patience >= 30:
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_rho


def main():
    print("="*70)
    print("V10 MULTIMODAL: DNABERT-2 + DeepFusion + Epigenetic Gating")
    print("="*70 + "\n")

    # Load data
    try:
        train_data, val_data, test_data = load_multimodal_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating synthetic data for testing...\n")

        np.random.seed(42)
        n_samples = 5000

        seqs = [''.join(np.random.choice(['A', 'T', 'G', 'C'], 30)) for _ in range(n_samples)]
        epis = np.random.randn(n_samples, 690).astype(np.float32)
        y = np.random.uniform(0.3, 0.9, n_samples).astype(np.float32)

        idx = np.random.permutation(n_samples)
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)

        train_data = ([seqs[i] for i in idx[:n_train]],
                     epis[idx[:n_train]], y[idx[:n_train]])
        val_data = ([seqs[i] for i in idx[n_train:n_train+n_val]],
                   epis[idx[n_train:n_train+n_val]], y[idx[n_train:n_train+n_val]])
        test_data = ([seqs[i] for i in idx[n_train+n_val:]],
                    epis[idx[n_train+n_val:]], y[idx[n_train+n_val:]])

    # Train ensemble
    models = []
    val_rhos = []

    n_models = 5

    for i in range(n_models):
        print(f"\n{'='*70}")
        print(f"MODEL {i+1}/{n_models} (DNABERT-2 Multimodal)")
        print(f"{'='*70}")

        model = DNABERTMultimodalV10()
        trained_model, val_rho = train_model(model, train_data, val_data, epochs=150, seed=i)

        models.append(trained_model)
        val_rhos.append(val_rho)

        torch.save(trained_model.state_dict(),
                  f'/Users/studio/Desktop/PhD/Proposal/models/multimodal_v10_seed{i}.pt')
        print(f"    Model {i+1}: Val Rho {val_rho:.4f}")

    # Ensemble evaluation
    print(f"\n{'='*70}")
    print("ENSEMBLE EVALUATION")
    print(f"{'='*70}")

    test_seqs, test_epis, test_y = test_data
    ensemble_pred = np.zeros(len(test_y))

    with torch.no_grad():
        for model in models:
            model.eval()
            test_epis_t = torch.FloatTensor(test_epis).to(device)
            params = model(test_seqs, test_epis_t)

            alpha, beta = params[:, 0], params[:, 1]
            pred = (alpha / (alpha + beta)).cpu().numpy()
            ensemble_pred += pred

    ensemble_pred /= len(models)
    ensemble_rho, ensemble_pval = spearmanr(test_y, ensemble_pred)

    print(f"\nIndividual Val Rho: {[f'{r:.4f}' for r in val_rhos]}")
    print(f"Mean: {np.mean(val_rhos):.4f} ± {np.std(val_rhos):.4f}")
    print(f"\nEnsemble Test Rho: {ensemble_rho:.4f}")
    print(f"Test p-value: {ensemble_pval:.2e}")
    print(f"Target: 0.911")
    print(f"Achievement: {ensemble_rho/0.911*100:.1f}%")

    if ensemble_rho >= 0.911:
        print("\n✅ TARGET ACHIEVED!")
    else:
        gap = 0.911 - ensemble_rho
        print(f"\n⚠️ Gap: {gap:.4f} ({gap/0.911*100:.1f}%)")

    # Save ensemble
    torch.save({
        'models': [m.state_dict() for m in models],
        'ensemble_pred': ensemble_pred,
        'val_rhos': val_rhos,
        'test_rho': ensemble_rho,
        'test_pval': ensemble_pval
    }, '/Users/studio/Desktop/PhD/Proposal/models/multimodal_v10_ensemble.pt')

    print(f"\n✓ Ensemble saved\n")


if __name__ == '__main__':
    main()
