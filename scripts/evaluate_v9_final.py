#!/usr/bin/env python3
"""
Final V9 Model Evaluation
Loads all trained models and computes Spearman rho and AUROC metrics
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cpu')

# ==================== MULTIMODAL ARCHITECTURE ====================

class TransformerSequenceEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Linear(4, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 30, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x.view(x.size(0), 30, 4)
        x = self.embed(x)
        x += self.pos_encoding
        x = self.transformer(x)
        x = self.ln(x)
        x = x.mean(dim=1)
        return x


class DeepFusion(nn.Module):
    def __init__(self, d_seq=256, d_epi=256, d_fusion=512):
        super().__init__()
        self.epi_proj = nn.Linear(690, d_epi)
        self.attn = nn.MultiheadAttention(d_fusion, num_heads=8, batch_first=True)
        self.seq_proj = nn.Linear(d_seq, d_fusion)
        self.epi_proj2 = nn.Linear(d_epi, d_fusion)
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
        seq_proj = self.seq_proj(seq_feat).unsqueeze(1)
        epi_proj = self.epi_proj2(epi_feat).unsqueeze(1)
        attn_out, _ = self.attn(seq_proj, epi_proj, epi_proj)
        combined = torch.cat([seq_proj.squeeze(1), attn_out.squeeze(1)], dim=1)
        fused = self.mlp(combined)
        return fused


class MultimodalV9(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_encoder = TransformerSequenceEncoder(d_model=256, n_layers=3)
        self.fusion = DeepFusion(d_seq=256, d_epi=256, d_fusion=512)
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, seq, epi):
        seq_feat = self.seq_encoder(seq)
        fused = self.fusion(seq_feat, epi)
        params = self.head(fused)
        alpha = torch.nn.functional.softplus(params[:, 0]) + 1e-3
        beta = torch.nn.functional.softplus(params[:, 1]) + 1e-3
        return alpha, beta


# ==================== OFF-TARGET ARCHITECTURE ====================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        class_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (class_weight * focal_weight * ce).mean()


class TransformerOffTarget(nn.Module):
    def __init__(self, d_model=128, n_heads=8, dropout=0.3):
        super().__init__()
        self.embed = nn.Linear(4, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 23, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), 23, 4)
        x = self.embed(x)
        x += self.pos_enc
        x = self.transformer(x)
        x = self.ln(x)
        x = x.mean(dim=1)
        logit = self.head(x).squeeze(-1)
        return logit


# ==================== DATA LOADING ====================

def load_multimodal_data():
    """Load multimodal test data from CSV."""
    data_dir = Path('/Users/studio/Desktop/PhD/Proposal/data/processed/split_a')
    test_dfs = []

    for cell_type in ['HCT116', 'HEK293T', 'HeLa']:
        test_path = data_dir / f'{cell_type}_test.csv'
        if test_path.exists():
            test_dfs.append(pd.read_csv(test_path))

    df = pd.concat(test_dfs, ignore_index=True)

    # Encode sequences (raw DNA → one-hot)
    def encode_seqs(seqs):
        mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0.25,0.25,0.25,0.25]}
        encoded = []
        for seq in seqs:
            code = []
            for base in str(seq)[:30]:  # Limit to 30bp
                code.extend(mapping.get(base.upper(), [0.25,0.25,0.25,0.25]))
            # Pad to 120 if shorter
            while len(code) < 120:
                code.extend([0.25,0.25,0.25,0.25])
            encoded.append(code[:120])
        return np.array(encoded, dtype=np.float32)

    seqs_encoded = encode_seqs(df['sequence'].values)

    # Get epigenomics features - pad to 690 dimensions
    epi_cols = [col for col in df.columns if col.startswith('feat_')]
    epi_data = df[epi_cols].values.astype(np.float32)
    # Pad to 690
    if epi_data.shape[1] < 690:
        epi_data = np.pad(epi_data, ((0,0), (0, 690-epi_data.shape[1])), mode='constant')
    epi_data = epi_data[:, :690]

    targets = df['efficiency'].values.astype(np.float32)

    return seqs_encoded, epi_data, targets, df


def load_offtarget_data():
    """Load off-target test data from TSV."""
    data_path = Path('/Users/studio/Desktop/PhD/Proposal/data/raw/crisprofft/CRISPRoffT_all_targets.txt')

    def one_hot_encode(seq, length=23):
        """One-hot encode DNA sequence."""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
        vec = np.zeros((length, 4), dtype=np.float32)
        seq = str(seq).upper()[:length]
        for i, c in enumerate(seq):
            if i < length and c in mapping:
                vec[i, mapping[c]] = 1
        return vec.flatten()

    seqs, labels = [], []

    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # Skip header

            parts = line.strip().split('\t')
            if len(parts) < 35:
                continue

            try:
                guide = parts[21]  # Column 21: Guide sequence
                target_status = parts[33] if len(parts) > 33 else "ON"  # Column 33: ON/OFF status

                if target_status not in ["ON", "OFF"]:
                    continue
                if not guide or len(guide) < 20:
                    continue

                # Label: 1 for OFF-target, 0 for ON-target
                label = 1.0 if target_status == "OFF" else 0.0
                seqs.append(guide)
                labels.append(label)
            except (ValueError, IndexError):
                continue

    labels = np.array(labels, dtype=np.float32)

    # Stratified split (70% train, 15% val, 15% test)
    np.random.seed(42)
    n = len(seqs)
    idx = np.random.permutation(n)

    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    test_idx = idx[n_train+n_val:]
    test_seqs = [seqs[i] for i in test_idx]
    test_labels = labels[test_idx]

    # Encode sequences using the same function as training
    X_test = np.array([one_hot_encode(s) for s in test_seqs]).astype(np.float32)

    return X_test, test_labels


# ==================== INFERENCE ====================

def evaluate_multimodal():
    """Evaluate multimodal V9 models."""
    print("\n" + "="*70)
    print("MULTIMODAL V9 EFFICACY PREDICTION")
    print("="*70)

    seqs, epis, targets, df = load_multimodal_data()
    print(f"✓ Loaded test data: {len(targets)} samples")
    print(f"  Efficiency range: [{targets.min():.4f}, {targets.max():.4f}]")

    seqs_t = torch.from_numpy(seqs).to(device)
    epis_t = torch.from_numpy(epis).to(device)

    all_preds = []

    for seed in range(5):
        model_path = Path(f'/Users/studio/Desktop/PhD/Proposal/models/multimodal_v9_seed{seed}.pt')
        if not model_path.exists():
            print(f"⚠️ Model seed {seed} not found")
            continue

        # Load model
        model = MultimodalV9().to(device).eval()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        # Infer
        with torch.no_grad():
            alpha, beta = model(seqs_t, epis_t)
            # Beta distribution mean: alpha / (alpha + beta)
            pred = alpha / (alpha + beta)
            all_preds.append(pred.cpu().numpy())

        # Single-model metrics
        single_rho, _ = spearmanr(targets, all_preds[-1])
        print(f"  Model {seed}: Rho = {single_rho:.4f}")

    # Ensemble prediction
    ensemble_pred = np.mean(all_preds, axis=0)
    ensemble_rho, p_val = spearmanr(targets, ensemble_pred)

    print(f"\n  ★ ENSEMBLE Spearman Rho: {ensemble_rho:.4f}")
    print(f"    P-value: {p_val:.2e}")
    print(f"    Target: 0.911")
    if ensemble_rho >= 0.911:
        print(f"    ✅ EXCEEDS TARGET by {(ensemble_rho-0.911)*100:.2f}%")
    else:
        gap = 0.911 - ensemble_rho
        print(f"    ⚠️  {gap*100:.2f}% below target")

    return ensemble_rho, p_val


def evaluate_offtarget():
    """Evaluate off-target V9 models."""
    print("\n" + "="*70)
    print("OFF-TARGET V9 CLASSIFICATION")
    print("="*70)

    seqs, labels = load_offtarget_data()
    print(f"✓ Loaded test data: {len(labels)} samples")
    print(f"  ON-target: {np.sum(labels)}, OFF-target: {len(labels)-np.sum(labels)}")
    print(f"  Ratio: {np.sum(labels)/(len(labels)-np.sum(labels)):.4f}:1")

    seqs_t = torch.from_numpy(seqs).to(device)

    all_probs = []

    for seed in range(20):
        model_path = Path(f'/Users/studio/Desktop/PhD/Proposal/models/off_target_v9_seed{seed}.pt')
        if not model_path.exists():
            print(f"⚠️ Model seed {seed} not found")
            continue

        # Load model
        model = TransformerOffTarget().to(device).eval()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        # Infer
        with torch.no_grad():
            logits = model(seqs_t)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

        # Single-model AUROC
        single_auroc = roc_auc_score(labels, probs)
        print(f"  Model {seed}: AUROC = {single_auroc:.4f}")

    # Ensemble prediction
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_auroc = roc_auc_score(labels, ensemble_probs)

    # Compute AP
    precision, recall, _ = precision_recall_curve(labels, ensemble_probs)
    ap = auc(recall, precision)

    print(f"\n  ★ ENSEMBLE AUROC: {ensemble_auroc:.4f}")
    print(f"    AP Score: {ap:.4f}")
    print(f"    Target: 0.99")
    if ensemble_auroc >= 0.99:
        print(f"    ✅ MEETS TARGET")
    else:
        gap = 0.99 - ensemble_auroc
        print(f"    ⚠️  {gap*100:.2f}% below target")

    return ensemble_auroc, ap


# ==================== MAIN ====================

def main():
    print("\n" + "="*70)
    print("V9 FINAL EVALUATION PIPELINE")
    print("="*70)

    results = {}

    # Multimodal
    try:
        rho, p_val = evaluate_multimodal()
        results['multimodal'] = {'spearman_rho': float(rho), 'p_value': float(p_val), 'target': 0.911}
    except Exception as e:
        print(f"❌ Multimodal evaluation failed: {e}")
        results['multimodal'] = {'error': str(e)}

    # Off-target
    try:
        auroc, ap = evaluate_offtarget()
        results['off_target'] = {'auroc': float(auroc), 'ap': float(ap), 'target': 0.99}
    except Exception as e:
        print(f"❌ Off-target evaluation failed: {e}")
        results['off_target'] = {'error': str(e)}

    # Save results
    results_file = Path('/Users/studio/Desktop/PhD/Proposal/logs/v9_evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {results_file}\n")


if __name__ == '__main__':
    main()
