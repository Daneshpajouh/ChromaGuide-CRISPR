import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import argparse
from typing import Tuple, List

def bootstrap_spearman(y_true: np.ndarray, y_pred: np.ndarray, n_resamples: int = 10000) -> Tuple[float, Tuple[float, float], np.ndarray]:
    """Compute Spearman rho with bootstrap confidence intervals."""
    rho_orig, _ = spearmanr(y_true, y_pred)
    rhos = []
    n = len(y_true)
    
    print(f"Running {n_resamples} bootstrap resamples...")
    for i in range(n_resamples):
        indices = np.random.choice(n, n, replace=True)
        r, _ = spearmanr(y_true[indices], y_pred[indices])
        rhos.append(r)
        if (i+1) % 2000 == 0:
            print(f"Completed {i+1} resamples")
            
    rhos = np.array(rhos)
    ci_low = np.percentile(rhos, 2.5)
    ci_high = np.percentile(rhos, 97.5)
    
    return rho_orig, (ci_low, ci_high), rhos

def paired_bootstrap_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray, n_resamples: int = 10000):
    """Paired bootstrap significance test for difference in Spearman rho."""
    rho_a, _ = spearmanr(y_true, y_pred_a)
    rho_b, _ = spearmanr(y_true, y_pred_b)
    delta_orig = rho_a - rho_b
    
    deltas = []
    n = len(y_true)
    
    print(f"Running {n_resamples} paired bootstrap resamples...")
    for i in range(n_resamples):
        indices = np.random.choice(n, n, replace=True)
        r_a, _ = spearmanr(y_true[indices], y_pred_a[indices])
        r_b, _ = spearmanr(y_true[indices], y_pred_b[indices])
        deltas.append(r_a - r_b)
        
    deltas = np.array(deltas)
    # p-value: proportion of resampled deltas that are <= 0 (if delta_orig > 0)
    if delta_orig > 0:
        p_val = np.mean(deltas <= 0)
    else:
        p_val = np.mean(deltas >= 0)
        
    return delta_orig, p_val, np.percentile(deltas, [2.5, 97.5])

def main():
    parser = argparse.ArgumentParser(description='ChromaGuide Bootstrap Significance Testing')
    parser.add_argument('--results_a', type=str, required=True, help='CSV file with predictions from model A')
    parser.add_argument('--results_b', type=str, help='CSV file with predictions from model B (for paired test)')
    parser.add_argument('--target_col', type=str, default='efficiency', help='Column name for ground truth')
    parser.add_argument('--pred_col', type=str, default='prediction', help='Column name for predictions')
    parser.add_argument('--n_resamples', type=int, default=10000, help='Number of bootstrap resamples')
    args = parser.parse_args()

    df_a = pd.read_csv(args.results_a)
    y_true = df_a[args.target_col].values
    y_pred_a = df_a[args.pred_col].values

    print(f"Model A: {args.results_a}")
    rho, (lo, hi), _ = bootstrap_spearman(y_true, y_pred_a, args.n_resamples)
    print(f"Spearman rho: {rho:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])")

    if args.results_b:
        df_b = pd.read_csv(args.results_b)
        # Ensure identical ordering
        if not all(df_a['sequence'] == df_b['sequence']):
            print("Warning: Sequences do not match. Attempting merge on sequence.")
            merged = pd.merge(df_a, df_b, on='sequence', suffixes=('_a', '_b'))
            y_true = merged[f'{args.target_col}_a'].values
            y_pred_a = merged[f'{args.pred_col}_a'].values
            y_pred_b = merged[f'{args.pred_col}_b'].values
        else:
            y_pred_b = df_b[args.pred_col].values

        print(f"\nModel B: {args.results_b}")
        delta, p_val, ci = paired_bootstrap_test(y_true, y_pred_a, y_pred_b, args.n_resamples)
        print(f"Delta Rho (A-B): {delta:.4f}")
        print(f"Bootstrap p-value: {p_val:.6f}")
        print(f"95% CI on Delta: [{ci[0]:.4f}, {ci[1]:.4f}]")
        
        if p_val < 0.001:
            print("Status: Statistically Significant (p < 0.001)")
        else:
            print(f"Status: Not Significant at alpha=0.001 (p={p_val:.6f})")

if __name__ == '__main__':
    main()
