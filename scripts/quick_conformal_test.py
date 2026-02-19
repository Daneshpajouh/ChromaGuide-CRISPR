#!/usr/bin/env python3
"""Quick Conformal Test - No plotting dependencies required."""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, ks_2samp
from transformers import AutoTokenizer, AutoModel
import sys
import json

# Add paths
sys.path.insert(0, '/home/amird/chromaguide_experiments/src')

def quick_conformal_test():
    """Quick conformal calibration test without plotting."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    model_path = "best_model_full.pt"
    data_path = "data/real/merged.csv"

    if not Path(model_path).exists():
        print(f"❌ Model {model_path} not found.")
        return False

    if not Path(data_path).exists():
        print(f"❌ Data {data_path} not found.")
        return False

    print("✅ Files found, starting quick test...")

    # Load data sample
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} samples")

    # Quick sample for testing
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)
    print(f"✅ Testing with {len(sample_df)} samples")

    # Split for conformal calibration
    cal_size = len(sample_df) // 2
    cal_df = sample_df.iloc[:cal_size]
    test_df = sample_df.iloc[cal_size:]

    print(f"✅ Split: {len(cal_df)} calibration, {len(test_df)} test")

    # Quick model test (simplified)
    try:
        # Try to load as full model first
        model = torch.load(model_path, map_location=device)
        print(f"✅ Loaded full model")
        model_type = "full"
    except:
        # Fallback to DNABERT + head loading
        try:
            MODEL_PATH = "zhihan1996/DNABERT-2-117M"
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            backbone = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)

            from chromaguide.prediction_head import BetaRegressionHead
            head = BetaRegressionHead(768)

            checkpoint = torch.load(model_path, map_location=device)
            backbone.load_state_dict(checkpoint['backbone_state_dict'])
            head.load_state_dict(checkpoint['head_state_dict'])

            model = {'backbone': backbone.to(device), 'head': head.to(device)}
            print(f"✅ Loaded DNABERT + Beta head")
            model_type = "dnabert_beta"
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

    # Quick conformity score simulation (for structure test)
    alpha = 0.1  # 90% coverage

    # Simulate conformity scores for testing
    np.random.seed(42)
    cal_scores = np.random.exponential(1.0, len(cal_df))
    test_scores = np.random.exponential(1.2, len(test_df))

    # Compute quantile
    quantile = np.quantile(cal_scores, 1 - alpha)
    print(f"✅ Computed {(1-alpha)*100}% quantile: {quantile:.4f}")

    # Simulate coverage
    coverage = np.mean(test_scores <= quantile)
    print(f"✅ Simulated coverage: {coverage:.3f} (target: 0.90)")

    # Test exchangeability
    ks_stat, ks_p = ks_2samp(cal_scores, test_scores)
    exchangeable = ks_p > 0.05
    print(f"✅ Exchangeability test: KS p-value = {ks_p:.4f} ({'✓' if exchangeable else '✗'})")

    # Spearman correlation test
    test_labels = test_df['efficiency'] if 'efficiency' in test_df.columns else test_df.iloc[:, 1]
    dummy_preds = np.random.uniform(0.1, 0.9, len(test_labels))
    spearman_rho, spearman_p = spearmanr(dummy_preds, test_labels)
    print(f"✅ Spearman ρ (dummy): {spearman_rho:.4f} (p={spearman_p:.4f})")

    # Summary
    results = {
        'model_loaded': True,
        'model_type': model_type,
        'data_samples': len(df),
        'test_samples': len(sample_df),
        'target_coverage': 1 - alpha,
        'simulated_coverage': float(coverage),
        'coverage_within_tolerance': abs(coverage - 0.9) <= 0.02,
        'exchangeability_satisfied': exchangeable,
        'ks_p_value': float(ks_p),
        'spearman_rho_dummy': float(spearman_rho)
    }

    print("\n" + "="*50)
    print("QUICK CONFORMAL TEST SUMMARY")
    print("="*50)
    print(f"Model Type: {model_type}")
    print(f"Data Loaded: {len(df):,} samples")
    print(f"Coverage Target: {(1-alpha)*100:.1f}%")
    print(f"Simulated Coverage: {coverage*100:.1f}% {'✓' if results['coverage_within_tolerance'] else '✗'}")
    print(f"Exchangeability: {'✓' if exchangeable else '✗'}")
    print("="*50)

    # Save results
    output_dir = Path('results/quick_conformal_test')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'quick_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to {output_dir}/quick_test_results.json")
    print("\n🎯 Ready for full conformal calibration!")

    return True

if __name__ == "__main__":
    try:
        success = quick_conformal_test()
        if success:
            print("✅ Quick test completed successfully!")
        else:
            print("❌ Quick test failed!")
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
