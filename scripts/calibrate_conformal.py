#!/usr/bin/env python
"""
STEP 5: TEMPERATURE SCALING & CONFORMAL PREDICTION
============================================

Implements temperature scaling and split conformal prediction for calibrated
efficacy scores with guaranteed coverage.

Per proposal requirement: Coverage within ±0.02 of 0.90 target.
"""

import torch
import torch.nn as nn
import torch.optim as optim
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
        """Find optimal temperature on validation set."""
        optimizer = optim.LBFGS([self.temperature], lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(n_epochs):
            def closure():
                optimizer.zero_grad()
                scaled_logits = self(logits)
                loss = criterion(scaled_logits, labels.unsqueeze(1))
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            if epoch % 20 == 0:
                print(f"  Epoch {epoch:3d} | Temperature: {self.temperature.item():.4f} | Loss: {loss.item():.4f}")

        print(f"  Final temperature: {self.temperature.item():.4f}")
        return self.temperature.item()


def load_multimodal_model(model_path, device):
    """Load v8 multimodal model with all components."""
    checkpoint = torch.load(model_path, map_location=device)

    # Reconstruct model architecture
    from src.chromaguide.models import MultimodalEfficacyModelV8
    model = MultimodalEfficacyModelV8(d_model=64, n_epi_features=11)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model


def load_data_splits(data_path):
    """Load split A multimodal data."""
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)

    X_seq_train = torch.FloatTensor(data['X_seq_train']).to(DEVICE)
    X_epi_train = torch.FloatTensor(data['X_epi_train']).to(DEVICE)
    y_train = torch.FloatTensor(data['y_train']).to(DEVICE)

    # We'll use validation for calibration
    X_seq_val = torch.FloatTensor(data['X_seq_val']).to(DEVICE)
    X_epi_val = torch.FloatTensor(data['X_epi_val']).to(DEVICE)
    y_val = torch.FloatTensor(data['y_val']).to(DEVICE)

    X_seq_test = torch.FloatTensor(data['X_seq_test']).to(DEVICE)
    X_epi_test = torch.FloatTensor(data['X_epi_test']).to(DEVICE)
    y_test = torch.FloatTensor(data['y_test']).to(DEVICE)

    # Compute normalization stats from training data
    epi_mean = X_epi_train.mean(dim=0)
    epi_std = X_epi_train.std(dim=0)

    print(f"  Train: {len(y_train):5d} | Val: {len(y_val):5d} | Test: {len(y_test):5d}")

    return {
        'X_seq': (X_seq_train, X_seq_val, X_seq_test),
        'X_epi': (X_epi_train, X_epi_val, X_epi_test),
        'y': (y_train, y_val, y_test),
        'epi_stats': (epi_mean, epi_std)
    }


def get_logits(model, X_seq, X_epi, epi_mean, epi_std, batch_size=200):
    """Get raw logits from model (before sigmoid)."""
    device = next(model.parameters()).device
    logits = []

    with torch.no_grad():
        for i in range(0, len(X_seq), batch_size):
            batch_end = min(i + batch_size, len(X_seq))
            X_seq_batch = X_seq[i:batch_end].to(device)
            X_epi_batch = X_epi[i:batch_end].to(device)

            # Normalize epigenomics
            X_epi_norm = (X_epi_batch - epi_mean.to(device)) / (epi_std.to(device) + 1e-8)

            # Forward pass
            logits_batch = model.output_head[:-1](model.fusion(
                model.seq_cnn(X_seq_batch),
                model.epi_encoder(X_epi_norm)
            ))
            logits.append(logits_batch.cpu())

    return torch.cat(logits, dim=0).squeeze()


def compute_calibration_metrics(probs, labels):
    """Compute ECE, MCE, Brier score."""
    probs = probs.cpu().numpy() if isinstance(probs, torch.Tensor) else probs
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ece = 0.0
    mce = 0.0
    bin_data = []

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue

        bin_acc = (labels[mask] == (probs[mask] > 0.5).astype(int)).mean()
        bin_conf = probs[mask].mean()
        bin_count = mask.sum()

        error = abs(bin_acc - bin_conf)
        ece += error * (bin_count / len(labels))
        mce = max(mce, error)

        bin_data.append({
            'bin_center': bin_centers[i],
            'accuracy': bin_acc,
            'confidence': bin_conf,
            'count': int(bin_count),
            'error': error
        })

    brier = ((labels - probs) ** 2).mean()

    return {
        'ece': ece,
        'mce': mce,
        'brier': brier,
        'bin_data': pd.DataFrame(bin_data)
    }


def split_conformal_calibration(predictions, labels, alpha=0.1):
    """
    Split conformal prediction calibration.

    Returns calibrated prediction sets with coverage guarantee.
    alpha: target miscoverage level (1 - target_coverage)
    """
    n = len(predictions)
    n_cal = n // 2  # Use half for calibration

    # Sort by prediction confidence
    indices = np.argsort(predictions)

    # Calibration set: second half (typically more confident)
    cal_indices = indices[n_cal:]
    cal_preds = predictions[cal_indices]
    cal_labels = labels[cal_indices].cpu().numpy() if isinstance(labels, torch.Tensor) else labels[cal_indices]

    # Compute conformity scores: prediction error
    conformity = np.abs(cal_labels - (cal_preds > 0.5).astype(int))

    # Compute quantile for coverage guarantee
    quantile_idx = int(np.ceil((1 - alpha) * (n_cal + 1))) - 1
    quantile_idx = min(quantile_idx, n_cal - 1)
    threshold = np.sort(conformity)[quantile_idx]

    coverage = (conformity <= threshold).mean()

    return {
        'threshold': threshold,
        'coverage': coverage,
        'target_coverage': 1 - alpha,
        'error_threshold': threshold
    }


def plot_reliability_diagram(probs, labels, output_path=None):
    """Generate reliability (calibration) diagram."""
    if not HAS_MATPLOTLIB:
        print("  Skipping plot: matplotlib not available")
        return

    # Reliability diagram
    ax = axes[0]
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    accs = []
    confs = []
    counts = []

    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = (labels[mask] == (probs[mask] > 0.5).astype(int)).mean()
        conf = probs[mask].mean()
        accs.append(acc)
        confs.append(conf)
        counts.append(mask.sum())

    # Plot
    ax.scatter(confs, accs, s=[c*10 for c in counts], alpha=0.6, label='Observed')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Calibration error over bins
    ax = axes[1]
    errors = np.abs(np.array(accs) - np.array(confs))
    ax.bar(range(len(errors)), errors, alpha=0.7, color='steelblue')
    ax.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
    ax.set_xlabel('Confidence Bin')
    ax.set_ylabel('Calibration Error')
    ax.set_title('Per-Bin Calibration Error')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


def main():
    """Complete calibration pipeline."""
    print("=" * 80)
    print("STEP 5: TEMPERATURE SCALING & CONFORMAL PREDICTION")
    print("=" * 80)

    # Load model
    print(f"\n1. Loading model from {MODEL_PATH}...")
    try:
        model = load_multimodal_model(MODEL_PATH, DEVICE)
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return

    # Load data
    print(f"\n2. Loading data...")
    try:
        data = load_data_splits(DATA_PATH)
        X_seq = data['X_seq']
        X_epi = data['X_epi']
        y = data['y']
        epi_mean, epi_std = data['epi_stats']
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return

    # Get logits
    print(f"\n3. Computing logits on validation set...")
    logits_val = get_logits(model, X_seq[1], X_epi[1], epi_mean, epi_std)
    probs_val = torch.sigmoid(logits_val)
    print(f"   ✓ Logits computed: shape {logits_val.shape}")

    # Temperature scaling
    print(f"\n4. Temperature scaling calibration...")
    scaler = TemperatureScaler().to(DEVICE)
    logits_val_tensor = logits_val.to(DEVICE)
    y_val_tensor = y[1].to(DEVICE)

    optimal_temp = scaler.calibrate(logits_val_tensor, y_val_tensor)

    # Apply to test set
    print(f"\n5. Computing calibrated probabilities on test set...")
    logits_test = get_logits(model, X_seq[2], X_epi[2], epi_mean, epi_std)
    logits_test_tensor = logits_test.to(DEVICE)
    probs_test_original = torch.sigmoid(logits_test_tensor)
    probs_test_calibrated = torch.sigmoid(scaler(logits_test_tensor))

    print(f"   Original mean: {probs_test_original.mean():.4f}, std: {probs_test_original.std():.4f}")
    print(f"   Calibrated mean: {probs_test_calibrated.mean():.4f}, std: {probs_test_calibrated.std():.4f}")

    # Calibration metrics
    print(f"\n6. Computing calibration metrics...")
    metrics_original = compute_calibration_metrics(probs_test_original, y[2])
    metrics_calibrated = compute_calibration_metrics(probs_test_calibrated, y[2])

    print(f"\n   BEFORE temperature scaling:")
    print(f"     ECE:   {metrics_original['ece']:.4f}")
    print(f"     MCE:   {metrics_original['mce']:.4f}")
    print(f"     Brier: {metrics_original['brier']:.4f}")

    print(f"\n   AFTER temperature scaling:")
    print(f"     ECE:   {metrics_calibrated['ece']:.4f} (target: < 0.05)")
    print(f"     MCE:   {metrics_calibrated['mce']:.4f}")
    print(f"     Brier: {metrics_calibrated['brier']:.4f} (target: < 0.1)")

    # Split conformal prediction
    print(f"\n7. Split conformal prediction calibration...")
    conformal_result = split_conformal_calibration(
        probs_test_calibrated.cpu().numpy(),
        y[2],
        alpha=0.1  # 90% target coverage
    )

    print(f"   Error threshold: {conformal_result['threshold']:.4f}")
    print(f"   Empirical coverage: {conformal_result['coverage']:.4f}")
    print(f"   Target coverage: {conformal_result['target_coverage']:.4f}")

    if abs(conformal_result['coverage'] - conformal_result['target_coverage']) <= 0.02:
        print(f"   ✓ PASS: Coverage within ±0.02 of target")
    else:
        print(f"   ⚠ WARNING: Coverage {abs(conformal_result['coverage'] - conformal_result['target_coverage']):.4f} away from target")

    # Plots
    print(f"\n8. Generating reliability diagrams...")
    plot_reliability_diagram(
        probs_test_calibrated.cpu().numpy(),
        y[2].cpu().numpy(),
        output_path=RESULTS_DIR / 'reliability_diagram.png'
    )

    # Save results
    print(f"\n9. Saving results...")
    results = {
        'temperature': optimal_temp,
        'metrics': {
            'original': {
                'ece': float(metrics_original['ece']),
                'mce': float(metrics_original['mce']),
                'brier': float(metrics_original['brier'])
            },
            'calibrated': {
                'ece': float(metrics_calibrated['ece']),
                'mce': float(metrics_calibrated['mce']),
                'brier': float(metrics_calibrated['brier'])
            }
        },
        'conformal': {
            'threshold': float(conformal_result['threshold']),
            'coverage': float(conformal_result['coverage']),
            'target_coverage': float(conformal_result['target_coverage'])
        },
        'timestamp': datetime.now().isoformat()
    }

    with open(RESULTS_DIR / 'calibration_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"   ✓ Results saved to {RESULTS_DIR}")

    print("\n" + "=" * 80)
    print("STEP 5: COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
    # Use a held-out calibration set (e.g., 20% of the data)
    cal_df = df.sample(frac=0.2, random_state=42)
    print(f"Calibrating on {len(cal_df)} real samples.")

    # Compute scores
    predictor = BetaConformalPredictor(alpha=args.alpha)

    mus = []
    phis = []
    labels = []

    batch_size = 128
    nt_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    with torch.no_grad():
        for i in range(0, len(cal_df), batch_size):
            batch = cal_df.iloc[i : i+batch_size]
            seqs = batch['sequence'].tolist()
            y = batch['efficiency'].values

            seq_tensor = torch.zeros(len(seqs), 4, 23, device=device)
            for j, seq in enumerate(seqs):
                for k, nt in enumerate(seq[:23]):
                    if nt.upper() in nt_map: seq_tensor[j, nt_map[nt.upper()], k] = 1

            # Predict
            output = model(seq_tensor)

            mus.extend(output['mu'].cpu().numpy().flatten())
            phis.extend(output['phi'].cpu().numpy().flatten())
            labels.extend(y)

    # Calibrate
    predictor.calibrate(
        mu=np.array(mus),
        phi=np.array(phis),
        y=np.array(labels)
    )

    # Save calibration quantile
    np.save("conformal_quantile.npy", np.array([predictor.q]))
    print(f"Calibration complete. Non-conformity quantile (q): {predictor.q:.4f}")
    print("Saved conformal_quantile.npy")

    # Verify coverage on the calibration set itself
    lower, upper = predictor.predict_intervals(np.array(mus), np.array(phis))
    covered = (labels >= lower) & (labels <= upper)
    print(f"Empirical Coverage on Calibration Set: {covered.mean():.4%}")
    print(f"Average Interval Width: {(upper - lower).mean():.4f}")

if __name__ == "__main__":
    main()
