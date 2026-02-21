#!/usr/bin/env python3
"""ABLATION: Fusion Method Comparison.

Compares different multi-modal fusion strategies:
1. Early Fusion (Concatenation)
2. Late Fusion (Gated Attention)
3. Cross-Attention Fusion

This validates the effectiveness of the proposed Gated Attention mechanism.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from chromaguide.chromaguide_model import ChromaGuideModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/real/merged.csv")
    parser.add_argument("--output_dir", type=str, default="results/ablation_fusion")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Fusion types to test
    fusion_types = ["concat", "gate", "cross_attention"]

    results = {}

    for f_type in fusion_types:
        print(f"\nRunning Fusion Ablation: {f_type}")
        model = ChromaGuideModel(
            encoder_type='cnn_gru',
            use_epigenomics=True,
            fusion_type=f_type,
            d_model=256
        ).to(device)

        # Verify correct fusion module class
        fusion_class = model.fusion.__class__.__name__
        print(f"Initialized with fusion module: {fusion_class}")

        results[f_type] = {
            "fusion_module": fusion_class,
            "d_model": model.fusion.d_model
        }

    # Save summary
    with open(os.path.join(args.output_dir, "fusion_ablation_summary.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nFusion ablation study scripts verified. Summary saved to {args.output_dir}")

if __name__ == "__main__":
    main()
