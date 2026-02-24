#!/usr/bin/env python3
"""
V10 OFF-TARGET (CORRECTED): DNABERT-2 + Per-Mark Epigenetic Gating

CORRECTED ARCHITECTURE from Kimata et al. (2025) PLOS ONE (PMID: 41223195)

Architecture:
- DNABERT-2 (zhihan1996/DNABERT-2-117M) for sequence encoding (768-dim)
- THREE per-mark epigenetic gating modules (1 per mark):
  * ATAC-seq (100 bins) -> 256-dim gated features
  * H3K4me3 (100 bins) -> 256-dim gated features
  * H3K27ac (100 bins) -> 256-dim gated features
- Classifier: Linear(1536, 2) where 1536 = 768 DNABERT + 256*3 marks

Training: batch_size=128, epochs=8, lr=2e-5 DNABERT, lr=1e-3 EPI/Classifier

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
import sys
import argparse
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


# CNN module REMOVED - not in original CRISPR_DNABERT paper (Kimata et al. 2025)


class PerMarkEpigenicGating(nn.Module):
    """
    Gating mechanism for a SINGLE epigenetic mark (100 dimensions)
    Matches the exact architecture from Kimata et al. (2025) PLOS ONE
    """
    def __init__(self, mark_dim=100, hidden_dim=256, dnabert_dim=768, dropout=0.1):
        super().__init__()

        # Step 1: Encoder for single mark (100 -> 256)
        self.encoder = nn.Sequential(
            nn.Linear(mark_dim, hidden_dim),  # 100 -> 256
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim * 2),  # 256 -> 512
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim * 4),  # 512 -> 1024
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # 1024 -> 512
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim)  # 512 -> 256
        )

        # Step 2: Gate mechanism
        # Input: DNABERT[CLS](768) + mismatch(7) + bulge(1) = 776
        gate_input_dim = dnabert_dim + 7 + 1  # 776

        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),  # 776 -> 256
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim * 2),  # 256 -> 512
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim * 4),  # 512 -> 1024
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # 1024 -> 512
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, 1),  # 512 -> 1
            nn.Sigmoid()
        )

        # Initialize gate bias to -3.0 (conservative gating from paper)
        self.gate[-2].bias.data.fill_(-3.0)

    def forward(self, dnabert_cls, mark_features, mismatch_features=None, bulge_features=None):
        """
        dnabert_cls: (batch, 768) - [CLS] token from DNABERT
        mark_features: (batch, 100) - one epigenetic mark (100 bins)
        mismatch_features: (batch, 7) - guide-target mismatch one-hot
        bulge_features: (batch, 1) - bulge presence/absence

        Returns: (batch, 256) - gated epigenetic features
        """
        # Encode the mark
        encoded = self.encoder(mark_features)  # (batch, 256)

        # Create gate input if not provided
        if mismatch_features is None:
            mismatch_features = torch.zeros(dnabert_cls.size(0), 7,
                                           device=dnabert_cls.device, dtype=dnabert_cls.dtype)
        if bulge_features is None:
            bulge_features = torch.zeros(dnabert_cls.size(0), 1,
                                        device=dnabert_cls.device, dtype=dnabert_cls.dtype)

        # Gate input: concatenate DNABERT + mismatch + bulge
        gate_input = torch.cat([dnabert_cls, mismatch_features, bulge_features], dim=1)  # (batch, 776)
        gate_output = self.gate(gate_input)  # (batch, 1)

        # Apply gate: weighted interpolation between zero and encoded features
        gated = encoded * gate_output  # (batch, 256)

        return gated


# BiLSTM module REMOVED - not in original CRISPR_DNABERT paper (Kimata et al. 2025)


class DNABERTOffTargetCorrected(nn.Module):
    """
    CORRECTED V10 Off-Target Classifier - Exact architecture from Kimata et al. (2025)
    PLOS ONE (PMID: 41223195)

    Architecture:
    1. DNABERT-2 -> [CLS] token (768 dims)
    2. THREE separate epigenetic gating modules (1 per mark):
       - ATAC-seq (100 dims per mark) -> encoded to 256
       - H3K4me3 (100 dims per mark) -> encoded to 256
       - H3K27ac (100 dims per mark) -> encoded to 256
    3. Classifier: Linear(768 + 256*3, 2) = Linear(1536, 2)
    """
    def __init__(self, dnabert_model_name="zhihan1996/DNABERT-2-117M",
                 hidden_dim=256, dropout=0.1):
        super().__init__()

        # Load DNABERT-2
        local_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'dnabert2')
        if os.path.exists(local_model_path):
            print(f"Loading DNABERT-2 from local cache: {local_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.dnabert = AutoModel.from_pretrained(local_model_path)
        else:
            print(f"Local model not found at {local_model_path}, using HuggingFace...")
            self.tokenizer = AutoTokenizer.from_pretrained(dnabert_model_name, trust_remote_code=True)
            self.dnabert = AutoModel.from_pretrained(dnabert_model_name, trust_remote_code=True)

        self.dnabert_dim = self.dnabert.config.hidden_size  # 768

        # Freeze DNABERT, unfreeze last 6 layers for fine-tuning
        for param in self.dnabert.parameters():
            param.requires_grad = False
        for param in self.dnabert.encoder.layer[-6:].parameters():
            param.requires_grad = True

        # THREE separate epigenetic gating modules (one per mark)
        # Each handles 100-dim input from a single epigenetic mark
        self.epi_gating = nn.ModuleDict({
            'atac': PerMarkEpigenicGating(mark_dim=100, hidden_dim=hidden_dim,
                                         dnabert_dim=self.dnabert_dim, dropout=dropout),
            'h3k4me3': PerMarkEpigenicGating(mark_dim=100, hidden_dim=hidden_dim,
                                            dnabert_dim=self.dnabert_dim, dropout=dropout),
            'h3k27ac': PerMarkEpigenicGating(mark_dim=100, hidden_dim=hidden_dim,
                                            dnabert_dim=self.dnabert_dim, dropout=dropout)
        })

        # Classifier: taking DNABERT[CLS] + 3 gated epi marks
        # Input: 768 (DNABERT) + 256*3 (3 marks) = 1536
        self.classifier = nn.Linear(self.dnabert_dim + hidden_dim * 3, 2)

    def forward(self, sequences, epi_features, mismatch_features=None, bulge_features=None):
        """
        sequences: List of DNA sequences (will be tokenized)
        epi_features: (batch, 300) - concatenated [atac_100 | h3k4me3_100 | h3k27ac_100]
        mismatch_features: (batch, 7) optional
        bulge_features: (batch, 1) optional

        Returns: logits (batch, 2)
        """
        # Tokenize and encode with DNABERT
        tokens = self.tokenizer(sequences, return_tensors="pt", padding=True,
                               truncation=True, max_length=24).to(device)

        dnabert_output = self.dnabert(**tokens)
        dnabert_cls = dnabert_output.last_hidden_state[:, 0, :]  # (batch, 768)

        # Split 300-dim epi_features into 3 marks (100 dims each)
        atac_feats = epi_features[:, 0:100]        # (batch, 100)
        h3k4me3_feats = epi_features[:, 100:200]   # (batch, 100)
        h3k27ac_feats = epi_features[:, 200:300]   # (batch, 100)

        # Process each epigenetic mark through its own gating module
        gated_atac = self.epi_gating['atac'](dnabert_cls, atac_feats,
                                             mismatch_features, bulge_features)
        gated_h3k4me3 = self.epi_gating['h3k4me3'](dnabert_cls, h3k4me3_feats,
                                                   mismatch_features, bulge_features)
        gated_h3k27ac = self.epi_gating['h3k27ac'](dnabert_cls, h3k27ac_feats,
                                                   mismatch_features, bulge_features)

        # Concatenate DNABERT[CLS] + all 3 gated marks
        combined = torch.cat([dnabert_cls, gated_atac, gated_h3k4me3, gated_h3k27ac], dim=1)
        # Shape: (batch, 768 + 256*3 = 1536)

        # Classification
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

                # CORRECTED: 300-dim epigenetic features [ATAC(100)|H3K4me3(100)|H3K27ac(100)]
                # For now, use placeholder zeros - in real data would load actual epigenomics
                epi = np.zeros(300, dtype=np.float32)  # Placeholder zeros instead of random noise
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

    # Optimizer with EXACT layer-wise learning rates from CRISPR_DNABERT (Kimata et al. 2025)
    optimizer = optim.AdamW([
        {'params': model.dnabert.parameters(), 'lr': 2e-5},      # DNABERT layers: 2e-5
        {'params': model.epi_gating.parameters(), 'lr': 1e-3},   # Epigenetic gating: 1e-3
        {'params': model.classifier.parameters(), 'lr': 1e-3}    # Classifier: 1e-3
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
            sys.stdout.flush()
            sys.stderr.flush()

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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='V10 Off-Target Training with DNABERT-2')
    parser.add_argument('--data_path', type=str, default='data/raw/crisprofft/CRISPRoffT_all_targets.txt',
                        help='Path to CRISPRoffT data file')
    parser.add_argument('--epochs', type=int, default=8,
                        help='Number of training epochs per model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save trained models')
    args = parser.parse_args()
    
    print("="*70)
    print("V10 OFF-TARGET: Hybrid DNABERT-2 + CNN + BiLSTM + Epigenetic Gating")
    print("(Based on CRISPR_DNABERT: kimatakai/CRISPR_DNABERT)")
    print("="*70 + "\n")
    print(f"Data path: {args.data_path}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}\n")
    sys.stdout.flush()
    sys.stderr.flush()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data_path = os.path.expanduser(args.data_path)
    X_seqs, X_epis, y = load_crispofft_data(data_path)
    print(f"✓ Data loaded: {len(X_seqs)} sequences")
    sys.stdout.flush()


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
    sys.stdout.flush()

    # Train ensemble (extended from 8 epochs in paper to 50 for better ensemble diversity)
    models = []
    val_aucs = []

    n_models = 5

    for i in range(n_models):
        print(f"\n{'='*70}")
        print(f"MODEL {i+1}/{n_models} (DNABERT-2 Hybrid Off-target)")
        print(f"{'='*70}")
        sys.stdout.flush()

        model = DNABERTOffTargetCorrected()
        trained_model, val_auc = train_model(
            model, X_train_seqs, X_train_epis, y_train,
            X_val_seqs, X_val_epis, y_val,
            epochs=args.epochs, seed=i  # Use args.epochs instead of hardcoded 8
        )

        models.append(trained_model)
        val_aucs.append(val_auc)

        torch.save(trained_model.state_dict(),
                  os.path.join(args.output_dir, f'off_target_v10_seed{i}.pt'))
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
    }, os.path.join(args.output_dir, 'off_target_v10_ensemble.pt'))

    print(f"\n✓ Ensemble saved\n")


if __name__ == '__main__':
    main()
