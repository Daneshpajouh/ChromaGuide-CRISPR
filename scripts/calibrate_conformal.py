"""Generate Conformal Calibration Scores for ChromaGuide.

This script takes a trained model and a calibration dataset (real data),
computes non-conformity scores, and saves them for the designer.
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoConfig
import sys
import os

from chromaguide.chromaguide_model import ChromaGuideModel
from chromaguide.conformal import BetaConformalPredictor

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    model_path = "best_model_full.pt"
    data_path = "/home/amird/chromaguide_experiments/data/real/merged.csv"
    MODEL_PATH = "zhihan1996/DNABERT-2-117M"

    if not Path(model_path).exists():
        print(f"Model {model_path} not found.")
        return

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # We need to recreate the head and backbone
    from chromaguide.prediction_head import BetaRegressionHead
    head = BetaRegressionHead(768)

    checkpoint = torch.load(model_path, map_location=device)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    head.load_state_dict(checkpoint['head_state_dict'])

    backbone.to(device).eval()
    head.to(device).eval()

    # Load Data
    df = pd.read_csv(data_path)
    # Use a held-out calibration set (e.g., 10% of the data)
    cal_df = df.sample(frac=0.1, random_state=42)
    print(f"Calibrating on {len(cal_df)} real samples.")

    # Compute scores
    predictor = BetaConformalPredictor(alpha=0.1)

    mus = []
    phis = []
    labels = []

    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(cal_df), batch_size):
            batch = cal_df.iloc[i : i+batch_size]
            seqs = batch['sequence'].tolist()
            y = batch['efficiency'].values

            tokens = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=30).to(device)
            outputs = backbone(tokens['input_ids'], tokens['attention_mask'])
            hidden = (outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]).mean(dim=1)

            # Predict
            mu, phi = head(hidden)

            mus.extend(mu.cpu().numpy().flatten())
            phis.extend(phi.cpu().numpy().flatten())
            labels.extend(y)

    # Calibrate
    predictor.calibrate(
        y_true=np.array(labels),
        mu_pred=np.array(mus),
        phi_pred=np.array(phis)
    )

    # Save calibration
    predictor.save_calibration("conformal_calibration.npy")
    print("Saved conformal_calibration.npy")

if __name__ == "__main__":
    main()
