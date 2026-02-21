"""Generate Conformal Calibration Scores for ChromaGuide.

This script takes a trained model and a calibration dataset (real data),
computes non-conformity scores, and saves them for the designer.
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import argparse

# Add src to path for ChromaGuide imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chromaguide.chromaguide_model import ChromaGuideModel
from chromaguide.conformal import BetaConformalPredictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="best_model_on_target.pt")
    parser.add_argument("--data_path", type=str, default="data/real/merged.csv")
    parser.add_argument("--backbone", type=str, default="cnn_gru")
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage rate (0.1 = 90% coverage)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(args.model_path):
        print(f"Model {args.model_path} not found.")
        return

    # Load Model
    model = ChromaGuideModel(
        encoder_type=args.backbone,
        d_model=256 if args.backbone == 'cnn_gru' else 768,
        use_epigenomics=False, # Standard seq-based calibration
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Load Data
    if not os.path.exists(args.data_path):
         # Local relative path fallback
        args.data_path = os.path.join(os.path.dirname(__file__), "../data/real/merged.csv")

    df = pd.read_csv(args.data_path)
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
