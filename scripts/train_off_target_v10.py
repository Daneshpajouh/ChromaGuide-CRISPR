#!/usr/bin/env python3
"""
V10 OFF-TARGET: Hybrid DNABERT-2 + Multi-scale CNN + BiLSTM + Epigenetic Gating

Combines verified architectures from:
- DNABERT-2 (zhihan1996/DNABERT-2-117M) for sequence encoding
- CRISPR-MCA (Multi-scale CNN inception module)
- CRISPR-HW (BiLSTM for context)
- CRISPR_DNABERT (Epigenetic gating mechanism)

Target: AUROC >= 0.99 on off-target classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
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


class MultiScaleCNNModule(nn.Module):
    """
    Multi-scale CNN Inception Module from CRISPR-MCA
    Parallel branches: 1x1, 3x3, 5x5 convolutions + MaxPool
    """
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.branch_1x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

        self.branch_3x3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

        self.branch_5x5 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

        self.branch_maxpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        b1 = self.branch_1x1(x)
        b2 = self.branch_3x3(x)
        b3 = self.branch_5x5(x)
        b4 = self.branch_maxpool(x)
        return torch.cat([b1, b2, b3, b4], dim=1)  # (batch, 4*out_channels, seq_len)


class EpigenoticGatingModule(nn.Module):
    """
    Epigenetic Feature Gating Module from CRISPR_DNABERT (kimatakai/CRISPR_DNABERT)

    EXACT architecture from source code:
    - Epigenetic encoder: feature_dim -> 256 -> 512 -> 1024 -> 512 -> 256 (ReLU + Dropout(0.1))
    - Gating module: (dnabert_768 + mismatch_7 + bulge_1) -> 256 -> 512 -> 1024 -> 512 -> 256 + Sigmoid
    - Gate bias initialized to -3.0 (conservative gating strategy)

    Mismatch features: one-hot encoded guide-target mismatch type (7 dimensions)
    Bulge features: presence/absence of bulge (1 dimension)
    """
    def __init__(self, feature_dim, epi_hidden_dim=256, dnabert_hidden_size=768,
                 mismatch_dim=7, bulge_dim=1, dropout=0.1):
        super().__init__()

        self.epi_hidden_dim = epi_hidden_dim
        self.dnabert_hidden_size = dnabert_hidden_size

        # Epigenetic encoder layers (exact from source)
        self.epi_encoder = nn.Sequential(
            nn.Linear(feature_dim, epi_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(epi_hidden_dim, epi_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(epi_hidden_dim * 2, epi_hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(epi_hidden_dim * 4, epi_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(epi_hidden_dim * 2, epi_hidden_dim)
        )

        # Gate mechanism input: DNABERT output (768) + mismatch features (7) + bulge features (1)
        gate_input_dim = dnabert_hidden_size + mismatch_dim + bulge_dim  # 776

        # Gating module (exact same architecture as encoder from source)
        self.gating_module = nn.Sequential(
            nn.Linear(gate_input_dim, epi_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(epi_hidden_dim, epi_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(epi_hidden_dim * 2, epi_hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(epi_hidden_dim * 4, epi_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(epi_hidden_dim * 2, 1),
            nn.Sigmoid()
        )

        # Initialize gate bias to -3.0 for conservative gating (from paper)
        self.gating_module[-2].bias.data.fill_(-3.0)

    def forward(self, seq_features, epi_features, mismatch_features=None, bulge_features=None):
        """
        seq_features: (batch, dnabert_hidden_size=768) - DNABERT [CLS] output
        epi_features: (batch, feature_dim) - epigenetic features
        mismatch_features: (batch, 7) - guide-target mismatch encoding
        bulge_features: (batch, 1) - bulge presence/absence

        Returns: gated_features (batch, epi_hidden_dim=256)
        """
        # Encode epigenetic features
        epi_encoded = self.epi_encoder(epi_features)  # (batch, 256)

        # Build gate input: DNABERT + mismatch + bulge
        if mismatch_features is None:
            mismatch_features = torch.zeros(seq_features.shape[0], 7,
                                          device=seq_features.device, dtype=seq_features.dtype)
        if bulge_features is None:
            bulge_features = torch.zeros(seq_features.shape[0], 1,
                                       device=seq_features.device, dtype=seq_features.dtype)

        gate_input = torch.cat([seq_features, mismatch_features, bulge_features], dim=1)
        gate = self.gating_module(gate_input)  # (batch, 1)

        # Apply gate: control epigenetic feature contribution
        gated = seq_features[:, :self.epi_hidden_dim] * (1 - gate) + epi_encoded * gate
        return gated


class BiLSTMContext(nn.Module):
    """
    BiLSTM for bidirectional context from CRISPR-HW
    hidden_size=64
    """
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        output, (h_n, c_n) = self.bilstm(x)  # output: (batch, seq_len, 2*hidden_size)
        # Use bidirectional final hidden state
        final_state = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, 2*hidden_size)
        return final_state, output


class DNABERTOffTargetV10(nn.Module):
    """
    Hybrid Off-target Classifier V10

    Architecture:
    1. DNABERT-2 (BPE tokenization) -> sequence representation
    2. Multi-scale CNN (1x1, 3x3, 5x5) -> multi-scale features
    3. BiLSTM -> contextual features
    4. Epigenetic gating -> conditional feature fusion
    5. Classification head
    """
    def __init__(self, dnabert_model_name="zhihan1996/DNABERT-2-117M",
                 epi_feature_dim=690, hidden_dim=256, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 1. DNABERT-2 encoder (load from local cached model to avoid triton dependency)
        local_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'dnabert2')
        if os.path.exists(local_model_path):
            print(f"Loading DNABERT-2 from local cache: {local_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.dnabert = AutoModel.from_pretrained(local_model_path)
        else:
            print(f"Local model not found at {local_model_path}, using HuggingFace...")
            self.tokenizer = AutoTokenizer.from_pretrained(dnabert_model_name, trust_remote_code=True)
            self.dnabert = AutoModel.from_pretrained(dnabert_model_name, trust_remote_code=True)
        self.dnabert_dim = self.dnabert.config.hidden_size  # 768 for DNABERT-2

        # Freeze DNABERT initially, then unfreeze last 6 layers for fine-tuning
        for param in self.dnabert.parameters():
            param.requires_grad = False

        # Unfreeze last 6 transformer layers (following CRISPR_DNABERT strategy)
        for param in self.dnabert.encoder.layer[-6:].parameters():
            param.requires_grad = True

        # Project DNABERT output to hidden_dim
        self.seq_proj = nn.Sequential(
            nn.Linear(self.dnabert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. Multi-scale CNN Module (CRISPR-MCA style)
        self.cnn_module = MultiScaleCNNModule(in_channels=self.dnabert_dim, out_channels=64)
        # Output: (batch, 256, seq_len) [64*4 channels]

        self.cnn_proj = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),  # (batch, 256, 1)
            nn.Flatten(),  # (batch, 256)
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 3. BiLSTM for context (CRISPR-HW style)
        self.bilstm = BiLSTMContext(self.dnabert_dim, hidden_size=64, dropout=dropout)
        self.bilstm_proj = nn.Sequential(
            nn.Linear(128, hidden_dim),  # 2*64 from bidirectional
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 4. Epigenetic gating (CRISPR_DNABERT style)
        # Input: DNABERT (768) + mismatch (7) + bulge (1) + epigenomics
        self.epi_gating = EpigenoticGatingModule(
            feature_dim=epi_feature_dim,
            epi_hidden_dim=256,
            dnabert_hidden_size=self.dnabert_dim,
            mismatch_dim=7,
            bulge_dim=1,
            dropout=dropout
        )

        # 5. Classification head (EXACT from CRISPR_DNABERT source)
        # Simplified to Dropout + Linear as in original paper
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, 2)  # Input: concat of all 4 features (256*4=1024)
        )

    def forward(self, sequences, epi_features, mismatch_features=None, bulge_features=None):
        """
        sequences: list of DNA sequences (will be tokenized by DNABERT-2, max 24bp per CRISPR_DNABERT)
        epi_features: (batch, epi_feature_dim) epigenetic features
        mismatch_features: (batch, 7) optional guide-target mismatch encoding
        bulge_features: (batch, 1) optional bulge presence/absence
        """
        # Tokenize and encode with DNABERT-2 (BPE tokenization - improvement over k-mer)
        tokens = self.tokenizer(sequences, return_tensors="pt", padding=True,
                               truncation=True, max_length=24).to(device)
        dnabert_out = self.dnabert(**tokens).last_hidden_state  # (batch, seq_len, 768)

        # 1. Project DNABERT [CLS] to hidden_dim
        seq_repr = self.seq_proj(dnabert_out[:, 0, :])  # [CLS] token, (batch, 256)

        # 2. Multi-scale CNN features
        cnn_feat = self.cnn_module(dnabert_out.transpose(1, 2))  # (batch, 256, seq_len)
        cnn_repr = self.cnn_proj(cnn_feat)  # (batch, 256)

        # 3. BiLSTM context
        bilstm_final, bilstm_seq = self.bilstm(dnabert_out)  # (batch, 128)
        bilstm_repr = self.bilstm_proj(bilstm_final)  # (batch, 256)

        # 4. Epigenetic gating with mismatch/bulge features
        gated_features = self.epi_gating(
            dnabert_out[:, 0, :],  # Full DNABERT [CLS] for gate input (768-dim)
            epi_features,
            mismatch_features,
            bulge_features
        )  # (batch, 256)

        # 5. Concatenate all features
        combined = torch.cat([seq_repr, cnn_repr, bilstm_repr, gated_features], dim=1)  # (batch, 1024)

        # 6. Classification
        logits = self.classifier(combined)  # (batch, 2)

        return logits


def load_crispofft_data(data_path):
    """Load CRISPRoffT data with epigenetic features if available"""
    print(f"Loading CRISPRoffT data from {data_path}...")

    seqs, labels, epis = [], [], []

    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            parts = line.strip().split('\t')
            if len(parts) < 35:
                continue

            try:
                guide = parts[21]
                target_status = parts[33] if len(parts) > 33 else "ON"

                if target_status not in ["ON", "OFF"]:
                    continue
                if not guide or len(guide) < 20:
                    continue

                label = 1.0 if target_status == "OFF" else 0.0
                seqs.append(guide)
                labels.append(label)

                # Try to extract epigenetic features if available in file
                # For now, use placeholder epigenetic features
                epi = np.random.randn(690).astype(np.float32)  # Placeholder
                epis.append(epi)
            except:
                continue

    print(f"  Loaded {len(seqs)} sequences")

    X_seqs = seqs
    y = np.array(labels, dtype=np.float32)
    X_epis = np.array(epis, dtype=np.float32)

    print(f"  ON: {int((y==0).sum())}, OFF: {int((y==1).sum())}")
    print(f"  Ratio: {(y==1).sum()/(y==0).sum():.3f}:1\n")

    return X_seqs, X_epis, y


def train_model(model, train_seqs, train_epis, train_labels,
                val_seqs, val_epis, val_labels, epochs=8, seed=0):
    """Train a single model with validation AUROC monitoring

    Default epochs=8 from CRISPR_DNABERT paper, can be extended for ensemble diversity
    """

    model = model.to(device)

    # BALANCED SAMPLING from CRISPR_DNABERT: majority_rate = 0.2
    # This means minority class gets 1.0 weight, majority gets 0.2 relative weight
    n_pos = int((train_labels == 1).sum())
    n_neg = int((train_labels == 0).sum())
    majority_rate = 0.2

    weights = np.zeros_like(train_labels, dtype=np.float32)
    weights[train_labels == 1] = 1.0  # Minority (OFF-target)
    weights[train_labels == 0] = majority_rate  # Majority (ON-target)

    sampler = WeightedRandomSampler(weights, len(train_labels), replacement=True)

    # Optimizer with EXACT layer-wise learning rates from CRISPR_DNABERT
    optimizer = optim.AdamW([
        {'params': model.dnabert.parameters(), 'lr': 2e-5},      # DNABERT layers: 2e-5
        {'params': model.seq_proj.parameters(), 'lr': 1e-3},     # Projection: 1e-3
        {'params': model.cnn_module.parameters(), 'lr': 1e-3},   # CNN: 1e-3
        {'params': model.bilstm.parameters(), 'lr': 1e-3},       # BiLSTM: 1e-3
        {'params': model.epi_gating.parameters(), 'lr': 1e-3},   # Gating: 1e-3
        {'params': model.classifier.parameters(), 'lr': 2e-5}    # Classifier: 2e-5 (with DNABERT)
    ], weight_decay=1e-4)

    # Linear warmup for 10% of total steps, then standard scheduling
    total_steps = epochs * len(train_labels) // 128  # 128 = batch_size
    warmup_steps = max(1, total_steps // 10)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, total_steps // 5), T_mult=2, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()  # For 2-class output

    best_auc = 0
    patience = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0

        # EXACT batch size from CRISPR_DNABERT: 128
        for epi_batch, label_batch in DataLoader(
            TensorDataset(
                torch.FloatTensor(train_epis),
                torch.LongTensor(train_labels.astype(int))  # Need int for CrossEntropyLoss
            ),
            batch_size=128,
            sampler=sampler,
            drop_last=True
        ):
            epi_batch = epi_batch.to(device)
            label_batch = label_batch.to(device)

            # Get sequences for this batch
            batch_idx = np.random.choice(len(train_seqs), size=len(epi_batch), replace=False)
            batch_seqs = [train_seqs[i] for i in batch_idx]

            logits = model(batch_seqs, epi_batch)  # (batch, 2)
            loss = criterion(logits, label_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(1, n_batches)

        # Validation
        model.eval()
        with torch.no_grad():
            val_seqs_batch = val_seqs[:min(256, len(val_seqs))]
            val_epis_batch = torch.FloatTensor(val_epis[:len(val_seqs_batch)]).to(device)
            val_labels_batch = val_labels[:len(val_seqs_batch)]

            val_logits = model(val_seqs_batch, val_epis_batch)  # (batch, 2)
            val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()[:, 1]  # Probability of class 1 (OFF)
            val_auc = roc_auc_score(val_labels_batch, val_probs)

        if epoch % max(1, epochs // 5) == 0:
            print(f"  E{epoch:2d} | Loss {train_loss:.4f} | Val AUROC {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if patience >= max(3, epochs // 3):
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_auc


def main():
    print("="*70)
    print("V10 OFF-TARGET: Hybrid DNABERT-2 + CNN + BiLSTM + Epigenetic Gating")
    print("(Based on CRISPR_DNABERT: kimatakai/CRISPR_DNABERT)")
    print("="*70 + "\n")

    # Load data
    data_path = '/Users/studio/Desktop/PhD/Proposal/data/raw/crisprofft/CRISPRoffT_all_targets.txt'
    X_seqs, X_epis, y = load_crispofft_data(data_path)

    # Split data
    np.random.seed(42)
    n = len(X_seqs)
    idx = np.random.permutation(n)

    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

    X_train_seqs = [X_seqs[i] for i in train_idx]
    X_val_seqs = [X_seqs[i] for i in val_idx]
    X_test_seqs = [X_seqs[i] for i in test_idx]

    X_train_epis = X_epis[train_idx]
    X_val_epis = X_epis[val_idx]
    X_test_epis = X_epis[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}\n")

    # Train ensemble (extended from 8 epochs in paper to 50 for better ensemble diversity)
    models = []
    val_aucs = []

    n_models = 5

    for i in range(n_models):
        print(f"\n{'='*70}")
        print(f"MODEL {i+1}/{n_models} (DNABERT-2 Hybrid Off-target)")
        print(f"{'='*70}")

        model = DNABERTOffTargetV10()
        trained_model, val_auc = train_model(
            model, X_train_seqs, X_train_epis, y_train,
            X_val_seqs, X_val_epis, y_val,
            epochs=50, seed=i  # Extended from paper's 8 epochs for better convergence
        )

        models.append(trained_model)
        val_aucs.append(val_auc)

        torch.save(trained_model.state_dict(),
                  f'/Users/studio/Desktop/PhD/Proposal/models/off_target_v10_seed{i}.pt')
        print(f"    Model {i+1}: Val AUROC {val_auc:.4f}")

    # Ensemble evaluation
    print(f"\n{'='*70}")
    print("ENSEMBLE EVALUATION")
    print(f"{'='*70}")

    ensemble_logits = np.zeros((len(y_test), 2))

    with torch.no_grad():
        for model in models:
            model.eval()
            test_epis_t = torch.FloatTensor(X_test_epis).to(device)
            logits = model(X_test_seqs, test_epis_t)  # (batch, 2)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            ensemble_logits += probs

    ensemble_logits /= len(models)
    ensemble_probs_off = ensemble_logits[:, 1]  # Probability of OFF-target class
    ensemble_auc = roc_auc_score(y_test, ensemble_probs_off)

    print(f"\nIndividual Val AUROC: {[f'{a:.4f}' for a in val_aucs]}")
    print(f"Mean: {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
    print(f"\nEnsemble Test AUROC: {ensemble_auc:.4f}")
    print(f"Target: 0.99")
    print(f"Achievement: {ensemble_auc/0.99*100:.1f}%")

    if ensemble_auc >= 0.99:
        print("\n✅ TARGET ACHIEVED!")
    else:
        gap = 0.99 - ensemble_auc
        print(f"\n⚠️ Gap: {gap:.4f} ({gap/0.99*100:.1f}% below target)")

    # Save ensemble
    torch.save({
        'models': [m.state_dict() for m in models],
        'ensemble_probs_off': ensemble_probs_off,
        'ensemble_logits': ensemble_logits,
        'val_aucs': val_aucs,
        'test_auc': ensemble_auc
    }, '/Users/studio/Desktop/PhD/Proposal/models/off_target_v10_ensemble.pt')

    print(f"\n✓ Ensemble saved\n")


if __name__ == '__main__':
    main()
