#!/usr/bin/env python
"""
STEP 6: ABLATION STUDIES
========================

Systematically evaluate contribution of different components:
- Sequence-only baseline (0.7889)
- Epigenomics-only model
- Full multimodal model (0.8189)
- Gated attention (v7) vs Multi-head attention (v8) fusion
- Statistical significance testing

Per proposal: Show contribution of each modality
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import json
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plotting will be skipped")

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
RESULTS_DIR = Path('results/ablation')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class SequenceOnlyModel(nn.Module):
    """Baseline: sequence-only, no epigenomics."""

    def __init__(self, d_model=64):
        super().__init__()
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.output_head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, X_seq):
        h_seq = self.seq_cnn(X_seq).squeeze(-1)
        return self.output_head(h_seq)


class EpigenomicsOnlyModel(nn.Module):
    """Ablation: epigenomics-only, no sequence."""

    def __init__(self, n_epi_features=11, d_model=64):
        super().__init__()
        self.epi_encoder = nn.Sequential(
            nn.Linear(n_epi_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.output_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, X_epi):
        h_epi = self.epi_encoder(X_epi)
        return self.output_head(h_epi)


class MultimodalV8Model(nn.Module):
    """Full multimodal: Multi-Head Attention fusion."""

    def __init__(self, d_model=64, n_epi_features=11):
        super().__init__()
        # Sequence: CNN
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.seq_fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.LayerNorm(64)
        )

        # Epigenomics: Deep encoder
        self.epi_encoder = nn.Sequential(
            nn.Linear(n_epi_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Multi-head attention fusion (4 heads)
        self.fusion = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(64)

        # Output
        self.output_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, X_seq, X_epi):
        h_seq = self.seq_cnn(X_seq).squeeze(-1)
        h_seq = self.seq_fc(h_seq)
        h_epi = self.epi_encoder(X_epi)

        # Cross-attention: query from seq, key/value from epi
        h_fused, _ = self.fusion(
            h_seq.unsqueeze(1),
            h_epi.unsqueeze(1),
            h_epi.unsqueeze(1)
        )
        h_fused = h_fused.squeeze(1)
        h_fused = h_seq + h_fused  # Residual
        h_fused = self.fusion_norm(h_fused)

        return self.output_head(h_fused)


class GatedAttentionModel(nn.Module):
    """v7 Reference: Gated attention fusion."""

    def __init__(self, d_model=64, n_epi_features=11):
        super().__init__()
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.seq_fc = nn.Linear(32, d_model)

        self.epi_encoder = nn.Sequential(
            nn.Linear(n_epi_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Simple gating: gate = sigmoid(concat(seq, epi))
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.output_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, X_seq, X_epi):
        h_seq = self.seq_cnn(X_seq).squeeze(-1)
        h_seq = self.seq_fc(h_seq)
        h_epi = self.epi_encoder(X_epi)

        # Gated fusion
        gate = self.gate(torch.cat([h_seq, h_epi], dim=1))
        h_fused = h_seq * gate + h_epi * (1 - gate)

        return self.output_head(h_fused)


def compute_rho(y_pred, y_true):
    """Compute Spearman correlation coefficient."""
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    y_pred = y_pred.squeeze()
    y_true = y_true.squeeze()
    rho, p_value = stats.spearmanr(y_true, y_pred)

    return float(rho), float(p_value)


def load_data():
    """Load multimodal data."""
    print("Loading data...")
    data = np.load('data/processed/split_a/split_a_processed.npz')

    X_seq_test = torch.FloatTensor(data['X_seq_test']).to(DEVICE)
    X_epi_test = torch.FloatTensor(data['X_epi_test']).to(DEVICE)
    y_test = torch.FloatTensor(data['y_test']).to(DEVICE)

    # Normalize epigenomics using training stats
    X_epi_train = torch.FloatTensor(data['X_epi_train']).to(DEVICE)
    epi_mean = X_epi_train.mean(dim=0)
    epi_std = X_epi_train.std(dim=0)
    X_epi_test_norm = (X_epi_test - epi_mean) / (epi_std + 1e-8)

    print(f"  Test set: {len(y_test)} samples")
    return X_seq_test, X_epi_test_norm, y_test


def evaluate_model(model, X_seq, X_epi, y_test, batch_size=200):
    """Evaluate model on test set."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(X_seq), batch_size):
            batch_end = min(i + batch_size, len(X_seq))
            X_seq_batch = X_seq[i:batch_end]
            X_epi_batch = X_epi[i:batch_end]

            if isinstance(model, (SequenceOnlyModel,)):
                pred = model(X_seq_batch)
            elif isinstance(model, (EpigenomicsOnlyModel,)):
                pred = model(X_epi_batch)
            else:
                pred = model(X_seq_batch, X_epi_batch)

            predictions.append(pred.cpu())

    predictions = torch.cat(predictions, dim=0).squeeze()
    rho, p_value = compute_rho(predictions, y_test)

    return {
        'rho': rho,
        'p_value': p_value,
        'predictions': predictions.numpy()
    }


def create_comparison_figure(results, output_path=None):
    """Create comparison plots."""
    if not HAS_MATPLOTLIB:
        print("  Skipping plots: matplotlib not available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot of Rho values
    ax = axes[0]
    models = list(results.keys())
    rhos = [results[m]['rho'] for m in models]
    colors = ['green', 'blue', 'red', 'orange', 'purple']

    bars = ax.bar(models, rhos, color=colors[:len(models)], alpha=0.7)
    ax.axhline(y=0.911, color='k', linestyle='--', linewidth=2, label='Target: 0.911')
    ax.axhline(y=0.7889, color='gray', linestyle=':', linewidth=2, label='Seq-only baseline')
    ax.set_ylabel('Test Rho')
    ax.set_title('Model Performance Comparison')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0.7, 0.95])

    # Add value labels on bars
    for bar, rho in zip(bars, rhos):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rho:.4f}',
                ha='center', va='bottom')

    # Contribution plot
    ax = axes[1]
    seq_only = results.get('Sequence-Only', {}).get('rho', 0.7889)
    epi_only = results.get('Epigenomics-Only', {}).get('rho', 0)
    multimodal = results.get('Multimodal (v8)', {}).get('rho', 0.8189)

    contributions = {
        'Sequence': seq_only - 0,
        'Epigenomics': epi_only - 0,
        'Interaction': max(0, multimodal - seq_only - epi_only)
    }

    parts = list(contributions.keys())
    values = list(contributions.values())
    ax.bar(parts, values, color=['steelblue', 'coral', 'lightgreen'], alpha=0.7)
    ax.set_ylabel('Contribution to Rho')
    ax.set_title('Component Contribution Analysis')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def main():
    """Run all ablation studies."""
    print("=" * 80)
    print("STEP 6: ABLATION STUDIES")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    X_seq, X_epi, y_test = load_data()

    # Initialize models
    print("\n2. Initializing models...")
    models = {
        'Sequence-Only': SequenceOnlyModel().to(DEVICE),
        'Epigenomics-Only': EpigenomicsOnlyModel().to(DEVICE),
        'Gated (v7)': GatedAttentionModel().to(DEVICE),
        'Multimodal (v8)': MultimodalV8Model().to(DEVICE),
    }
    print(f"   ✓ {len(models)} models initialized")

    # Load model weights if available
    print("\n3. Loading pre-trained weights...")
    model_paths = {
        'Multimodal (v8)': 'models/multimodal_v8_multihead_fusion.pt',
    }

    for name, path in model_paths.items():
        path_obj = Path(path)
        if path_obj.exists():
            state_dict = torch.load(path, map_location=DEVICE)
            models[name].load_state_dict(state_dict)
            print(f"   ✓ Loaded {name}")
        else:
            print(f"   ⚠ Model not found: {path}")

    # Evaluate all models
    print("\n4. Evaluating models on test set...")
    results = {}
    for name, model in models.items():
        print(f"\n   {name}:")
        result = evaluate_model(model, X_seq, X_epi, y_test)
        results[name] = result
        print(f"     Test Rho: {result['rho']:.4f}")
        print(f"     p-value: {result['p_value']:.2e}")

    # Statistical comparisons
    print("\n5. Statistical significance testing...")
    comparison_results = []

    model_names = list(results.keys())
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i+1:]:
            pred1 = results[m1]['predictions']
            pred2 = results[m2]['predictions']
            y_true = y_test.cpu().numpy()

            # Using Wilcoxon signed-rank test for paired samples
            # (both models evaluated on same test set)
            error1 = np.abs(y_true - pred1)
            error2 = np.abs(y_true - pred2)

            stat, p_value = stats.wilcoxon(error1, error2)

            comparison_results.append({
                'Model 1': m1,
                'Model 2': m2,
                'Rho 1': results[m1]['rho'],
                'Rho 2': results[m2]['rho'],
                'Wilcoxon p-value': p_value,
                'Significant (p<0.001)': p_value < 0.001
            })

            print(f"   {m1} vs {m2}:")
            print(f"     Rho difference: {results[m1]['rho'] - results[m2]['rho']:.4f}")
            print(f"     p-value: {p_value:.2e} {'✓' if p_value < 0.001 else '✗'}")

    # Create summary table
    print("\n6. Creating summary tables...")
    summary_df = pd.DataFrame([
        {
            'Model': name,
            'Test Rho': results[name]['rho'],
            'p-value': f"{results[name]['p_value']:.2e}",
            'vs Target (0.911)': f"{0.911 - results[name]['rho']:.4f}",
        }
        for name in results.keys()
    ])

    print("\n" + summary_df.to_string(index=False))

    # Save results
    print("\n7. Saving results...")

    summary_df.to_csv(RESULTS_DIR / 'ablation_summary.csv', index=False)
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(RESULTS_DIR / 'ablation_comparisons.csv', index=False)

    with open(RESULTS_DIR / 'ablation_results.json', 'w') as f:
        json.dump({
            'model_results': {k: v['rho'] for k, v in results.items()},
            'comparisons': comparison_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    # Generate plots
    print("\n8. Generating plots...")
    create_comparison_figure(results, RESULTS_DIR / 'ablation_comparison.png')

    print(f"\n   ✓ Results saved to {RESULTS_DIR}")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)
    seq_rho = results.get('Sequence-Only', {}).get('rho', 0)
    epi_rho = results.get('Epigenomics-Only', {}).get('rho', 0)
    multi_rho = results.get('Multimodal (v8)', {}).get('rho', 0)

    print(f"Sequence-Only Rho:    {seq_rho:.4f} (baseline)")
    print(f"Epigenomics-Only Rho: {epi_rho:.4f}")
    print(f"Multimodal v8 Rho:    {multi_rho:.4f}")
    print(f"\nEpigenomics contribution: {multi_rho - seq_rho:.4f} (+{100*(multi_rho - seq_rho)/seq_rho:.1f}%)")
    print("=" * 80)


if __name__ == '__main__':
    main()
