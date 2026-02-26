"""Evaluation metrics for ChromaGuide.

Implements all metrics from the proposal:
    Primary: Spearman ρ, Pearson r
    Regression: MSE, MAE, R²
    Ranking: nDCG@K, Precision@K
    Calibration: ECE, Brier score, CRPS
    Uncertainty: Coverage, interval width
    Off-target: AUROC, AUPRC, F1
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score, f1_score,
    brier_score_loss,
)
from typing import Optional


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    phi_pred: np.ndarray | None = None,
    metric_types: list[str] = ["primary", "secondary", "ranking"],
) -> dict[str, float]:
    """Compute all evaluation metrics.
    
    Args:
        y_true: True values (n,).
        y_pred: Predicted values (n,).
        phi_pred: Predicted precision from Beta regression (n,).
        metric_types: Which groups of metrics to compute.
    
    Returns:
        Dict of metric_name → value.
    """
    results = {}
    
    if "primary" in metric_types:
        rho, rho_p = spearmanr(y_true, y_pred)
        results["spearman_rho"] = float(rho)
        results["spearman_p"] = float(rho_p)
    
    if "secondary" in metric_types or "primary" in metric_types:
        r, r_p = pearsonr(y_true, y_pred)
        results["pearson_r"] = float(r)
        results["mse"] = float(mean_squared_error(y_true, y_pred))
        results["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        results["mae"] = float(mean_absolute_error(y_true, y_pred))
        results["r2"] = float(r2_score(y_true, y_pred))
    
    if "ranking" in metric_types:
        for k in [5, 10, 20]:
            results[f"ndcg_{k}"] = float(ndcg_at_k(y_true, y_pred, k))
            results[f"precision_{k}"] = float(precision_at_k(y_true, y_pred, k))
    
    if "calibration" in metric_types:
        results["ece"] = float(expected_calibration_error(y_true, y_pred))
        results["brier"] = float(brier_score_loss(
            (y_true > np.median(y_true)).astype(int),
            y_pred,
        ))
        if phi_pred is not None:
            results["crps"] = float(continuous_ranked_probability_score(
                y_true, y_pred, phi_pred
            ))
    
    return results


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Normalized Discounted Cumulative Gain at K.
    
    Measures whether the model ranks the top-K most effective
    sgRNAs correctly.
    """
    # Get top-K predicted indices
    pred_top_k = np.argsort(y_pred)[::-1][:k]
    
    # DCG of predicted ranking
    dcg = sum(
        y_true[pred_top_k[i]] / np.log2(i + 2)
        for i in range(k)
    )
    
    # Ideal DCG (perfect ranking)
    ideal_top_k = np.argsort(y_true)[::-1][:k]
    idcg = sum(
        y_true[ideal_top_k[i]] / np.log2(i + 2)
        for i in range(k)
    )
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int, threshold: float = 0.7) -> float:
    """Precision at K: fraction of top-K predictions that are truly effective.
    
    An sgRNA is "truly effective" if y_true > threshold (default 0.7).
    """
    pred_top_k = np.argsort(y_pred)[::-1][:k]
    truly_effective = y_true[pred_top_k] > threshold
    return truly_effective.mean()


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE).
    
    For regression: bins predictions and checks if average prediction
    matches average target within each bin.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    
    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        
        bin_pred_mean = y_pred[mask].mean()
        bin_true_mean = y_true[mask].mean()
        bin_weight = mask.sum() / total
        
        ece += bin_weight * abs(bin_pred_mean - bin_true_mean)
    
    return ece


def continuous_ranked_probability_score(
    y_true: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
    n_samples: int = 1000,
) -> float:
    """Continuous Ranked Probability Score (CRPS) for Beta predictions.
    
    CRPS measures the quality of probabilistic predictions.
    Lower is better.
    """
    from scipy.stats import beta as beta_dist
    
    crps_values = []
    for i in range(len(y_true)):
        a = mu[i] * phi[i]
        b = (1 - mu[i]) * phi[i]
        
        if a <= 0 or b <= 0:
            crps_values.append(abs(y_true[i] - mu[i]))
            continue
        
        # Approximate CRPS via sampling
        samples = beta_dist.rvs(a, b, size=n_samples)
        crps = np.mean(np.abs(samples - y_true[i])) - 0.5 * np.mean(
            np.abs(samples[:n_samples//2] - samples[n_samples//2:])
        )
        crps_values.append(max(0, crps))
    
    return np.mean(crps_values)


def compute_offtarget_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute off-target prediction metrics."""
    results = {}
    
    try:
        results["auroc"] = float(roc_auc_score(y_true, y_pred))
    except ValueError:
        results["auroc"] = 0.5
    
    try:
        results["auprc"] = float(average_precision_score(y_true, y_pred))
    except ValueError:
        results["auprc"] = 0.0
    
    y_pred_binary = (y_pred > threshold).astype(int)
    results["f1"] = float(f1_score(y_true, y_pred_binary, zero_division=0))
    
    return results
