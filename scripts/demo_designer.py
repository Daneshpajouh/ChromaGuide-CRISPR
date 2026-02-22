
import os
import sys
import torch
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chromaguide.designer import ChromaGuideDesigner
from chromaguide.chromaguide_model import ChromaGuideModel

def main():
    print("DEMO: ChromaGuide Designer Score S")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # 1. Create On-target checkpoint if not exists
    on_ckpt = "best_model_on_target.pt"
    if not os.path.exists(on_ckpt):
        model = ChromaGuideModel(encoder_type='cnn_gru', d_model=256, seq_len=21, num_epi_tracks=11, use_epigenomics=True).to(device)
        torch.save(model.state_dict(), on_ckpt)
        print(f"Created temporary on-target checkpoint: {on_ckpt}")

    # 2. Create Off-target checkpoint
    from chromaguide.off_target import OffTargetScorer
    off_ckpt = "dummy_off_target.pt"
    off_model = OffTargetScorer().to(device)
    torch.save(off_model.state_dict(), off_ckpt)
    print(f"Created temporary off-target checkpoint: {off_ckpt}")

    # 3. Initialize Designer (matching signature in src/chromaguide/designer.py)
    designer = ChromaGuideDesigner(
        on_target_checkpoint=on_ckpt,
        off_target_checkpoint=off_ckpt,
        device=device
    )

    # Dummy calibration to avoid "Predictor must be calibrated before use."
    print("Performing dummy calibration of Conformal Predictor...")
    import numpy as np
    dummy_mu = np.random.uniform(0.1, 0.9, 100)
    dummy_phi = np.random.uniform(5, 50, 100)
    dummy_y = np.random.uniform(0.1, 0.9, 100)
    designer.conformal.calibrate(dummy_mu, dummy_phi, dummy_y)

    # Sample DNA region (containing NGG sites)
    # Each sequence must be 23nt for the designer's internal search
    target_dna = "ACCCCCCCCCTGTTCCCCTTTGG" + "G" * 30 + "GGGGGGGGGGGGGGGGGGGGGGG"

    # Epigenomic features (11 dims)
    # For now, designer.design_guides doesn't fully map features per position yet.
    # It passes epi_features directly to the model.
    epi_features = torch.zeros(1, 11, 1).to(device)

    print(f"\nScanning DNA region for Designer Score S...")
    results_df = designer.design_guides(target_dna, epi_features=epi_features)

    # Ensure Designer Score S is present (the 3rd fix)
    if 'designer_score_S' in results_df.columns:
        print("✓ SUCCESS: Designer Score S found in output reporting.")
    else:
        print("✗ ERROR: Designer Score S missing.")

    print("\nResults Preview:")
    print(results_df[['sequence', 'efficiency_mu', 'designer_score_S', 'uncertainty_sigma']])

    output_path = "designer_demo_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nDemo results saved to {output_path}")

if __name__ == "__main__":
    main()
