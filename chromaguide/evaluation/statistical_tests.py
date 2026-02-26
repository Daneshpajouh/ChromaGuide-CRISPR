"""Statistical tests for model comparison.

Implements the statistical testing framework from the proposal:
    1. 5×2cv paired t-test (primary model comparison)
    2. BCa bootstrap confidence intervals (10,000 resamples)
    3. Holm-Bonferroni correction (FWER for primary hypotheses H1-H3)
    4. Benjamini-Hochberg correction (FDR for ablation studies)
"""
from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def five_by_two_cv_paired_t_test(
    scores_a: list[list[float]],
    scores_b: list[list[float]],
) -> dict[str, float]:
    """5×2 cross-validated paired t-test.
    
    From Dietterich (1998): "Approximate Statistical Tests for Comparing 
    Supervised Classification Learning Algorithms"
    
    Args:
        scores_a: 5×2 array of performance scores for model A.
                  scores_a[i][j] = score for repeat i, fold j.
        scores_b: Same for model B.
    
    Returns:
        Dict with t-statistic and p-value.
    """
    assert len(scores_a) == 5 and len(scores_b) == 5
    assert all(len(s) == 2 for s in scores_a)
    assert all(len(s) == 2 for s in scores_b)
    
    # Compute differences for each fold
    p_values = []
    variances = []
    
    for i in range(5):
        d1 = scores_a[i][0] - scores_b[i][0]
        d2 = scores_a[i][1] - scores_b[i][1]
        
        d_mean = (d1 + d2) / 2
        d_var = (d1 - d_mean) ** 2 + (d2 - d_mean) ** 2
        
        variances.append(d_var)
    
    # First fold, first repeat difference
    d_11 = scores_a[0][0] - scores_b[0][0]
    
    # t-statistic
    variance_sum = sum(variances)
    if variance_sum == 0:
        return {"t_statistic": 0.0, "p_value": 1.0, "significant": False}
    
    t_stat = d_11 / np.sqrt(variance_sum / 5)
    
    # p-value (two-tailed, 5 df)
    p_value = 2 * stats.t.sf(abs(t_stat), df=5)
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "df": 5,
    }


def bca_bootstrap_ci(
    metric_fn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bias-corrected and accelerated (BCa) bootstrap confidence intervals.
    
    Args:
        metric_fn: Function(y_true, y_pred) → scalar metric.
        y_true: True values (n,).
        y_pred: Predicted values (n,).
        n_resamples: Number of bootstrap resamples.
        confidence_level: CI level (default 0.95 = 95% CI).
    
    Returns:
        Dict with point estimate, lower, upper bounds.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    
    # Point estimate
    point_estimate = metric_fn(y_true, y_pred)
    
    # Bootstrap distribution
    bootstrap_stats = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        try:
            stat = metric_fn(y_true[idx], y_pred[idx])
            bootstrap_stats.append(stat)
        except Exception:
            continue
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    if len(bootstrap_stats) < 100:
        logger.warning("Too few successful bootstrap resamples")
        return {
            "point_estimate": float(point_estimate),
            "lower": float(point_estimate),
            "upper": float(point_estimate),
            "std": 0.0,
        }
    
    # BCa correction
    # Bias correction factor
    z0 = stats.norm.ppf(np.mean(bootstrap_stats < point_estimate))
    
    # Acceleration factor (jackknife)
    jackknife_stats = []
    for i in range(n):
        idx = np.concatenate([np.arange(i), np.arange(i + 1, n)])
        try:
            stat = metric_fn(y_true[idx], y_pred[idx])
            jackknife_stats.append(stat)
        except Exception:
            jackknife_stats.append(point_estimate)
    
    jackknife_stats = np.array(jackknife_stats)
    jack_mean = jackknife_stats.mean()
    numerator = np.sum((jack_mean - jackknife_stats) ** 3)
    denominator = 6 * (np.sum((jack_mean - jackknife_stats) ** 2)) ** 1.5
    
    if denominator == 0:
        a = 0.0
    else:
        a = numerator / denominator
    
    # Adjusted percentiles
    alpha = 1 - confidence_level
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)
    
    def adjust_percentile(z_alpha):
        num = z0 + z_alpha
        denom = 1 - a * num
        if denom == 0:
            return stats.norm.cdf(z_alpha)
        adjusted = stats.norm.cdf(z0 + num / denom)
        return np.clip(adjusted, 0.001, 0.999)
    
    lower_pct = adjust_percentile(z_alpha_lower)
    upper_pct = adjust_percentile(z_alpha_upper)
    
    lower = float(np.quantile(bootstrap_stats, lower_pct))
    upper = float(np.quantile(bootstrap_stats, upper_pct))
    
    return {
        "point_estimate": float(point_estimate),
        "lower": lower,
        "upper": upper,
        "std": float(bootstrap_stats.std()),
        "n_resamples": len(bootstrap_stats),
    }


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Holm-Bonferroni step-down procedure for FWER control.
    
    Used for primary hypotheses (H1, H2, H3).
    
    Args:
        p_values: List of p-values.
        alpha: Family-wise error rate.
    
    Returns:
        List of dicts with adjusted p-values and significance.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    
    results = [None] * n
    
    for rank, idx in enumerate(sorted_indices):
        adjusted_alpha = alpha / (n - rank)
        p = p_values[idx]
        
        results[idx] = {
            "original_p": float(p),
            "adjusted_alpha": float(adjusted_alpha),
            "significant": p < adjusted_alpha,
            "rank": rank + 1,
        }
    
    return results


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Benjamini-Hochberg procedure for FDR control.
    
    Used for ablation studies.
    
    Args:
        p_values: List of p-values.
        alpha: False discovery rate.
    
    Returns:
        List of dicts with adjusted p-values and significance.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    
    results = [None] * n
    
    # Find the largest k such that p_(k) ≤ k/n * α
    max_significant_rank = 0
    for rank, idx in enumerate(sorted_indices):
        adjusted_threshold = (rank + 1) / n * alpha
        if p_values[idx] <= adjusted_threshold:
            max_significant_rank = rank + 1
    
    for rank, idx in enumerate(sorted_indices):
        p = p_values[idx]
        adjusted_p = min(1.0, p * n / (rank + 1))
        
        results[idx] = {
            "original_p": float(p),
            "adjusted_p": float(adjusted_p),
            "significant": (rank + 1) <= max_significant_rank,
            "rank": rank + 1,
        }
    
    return results


def run_statistical_tests(
    model_results: dict[str, dict],
    baseline_name: str = "chromecrispr",
    alpha: float = 0.05,
) -> dict:
    """Run the full statistical testing battery.
    
    Args:
        model_results: Dict mapping model_name → results dict with:
            - cv_scores: 5×2 CV scores
            - test_predictions: (y_true, y_pred) arrays
        baseline_name: Name of baseline model for comparison.
    
    Returns:
        Comprehensive statistical analysis results.
    """
    from scipy.stats import spearmanr
    
    results = {
        "model_comparisons": {},
        "bootstrap_cis": {},
        "hypothesis_tests": {},
    }
    
    baseline = model_results.get(baseline_name)
    
    for model_name, model_data in model_results.items():
        if model_name == baseline_name:
            continue
        
        # 1. 5×2cv paired t-test
        if "cv_scores" in model_data and baseline and "cv_scores" in baseline:
            comparison = five_by_two_cv_paired_t_test(
                model_data["cv_scores"],
                baseline["cv_scores"],
            )
            results["model_comparisons"][f"{model_name}_vs_{baseline_name}"] = comparison
        
        # 2. BCa bootstrap CI
        if "test_predictions" in model_data:
            y_true, y_pred = model_data["test_predictions"]
            
            def spearman_metric(yt, yp):
                return spearmanr(yt, yp)[0]
            
            ci = bca_bootstrap_ci(
                spearman_metric, y_true, y_pred,
                n_resamples=10000,
            )
            results["bootstrap_cis"][model_name] = ci
    
    # 3. Multiple testing correction
    if results["model_comparisons"]:
        p_values = [v["p_value"] for v in results["model_comparisons"].values()]
        
        # Primary hypotheses: Holm-Bonferroni
        results["hypothesis_tests"]["holm_bonferroni"] = holm_bonferroni(p_values, alpha)
        
        # Ablations: Benjamini-Hochberg
        results["hypothesis_tests"]["benjamini_hochberg"] = benjamini_hochberg(p_values, alpha)
    
    return results
