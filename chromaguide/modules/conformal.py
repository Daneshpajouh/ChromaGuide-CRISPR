"""Split Conformal Prediction for CRISPR efficacy.

Provides distribution-free prediction intervals with finite-sample
coverage guarantees:
    P(Y ∈ C(X)) ≥ 1 - α  for all distributions P.

Uses the Beta regression variance σ²(x) = μ(1-μ)/(1+φ) as the
normalization function for Conformalized Quantile Regression (CQR).

Coverage target: 90% (α = 0.10), with tolerance ±0.02.
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Optional


class SplitConformalPredictor:
    """Split conformal predictor with normalized residuals.
    
    Algorithm:
        1. On calibration set:
           - Compute residuals r_i = |y_i - μ̂(x_i)|
           - Normalize: s_i = r_i / σ̂(x_i)  (using Beta variance)
           - Find q̂ = Quantile(s_1,...,s_n; (1-α)(1+1/n))
        2. On test set:
           - C(x) = [μ̂(x) - q̂·σ̂(x), μ̂(x) + q̂·σ̂(x)]
    """
    
    def __init__(
        self,
        alpha: float = 0.10,
        use_beta_sigma: bool = True,
        tolerance: float = 0.02,
    ):
        """
        Args:
            alpha: Target miscoverage rate (0.10 = 90% coverage).
            use_beta_sigma: If True, normalize residuals by Beta variance.
            tolerance: Acceptable deviation from target coverage.
        """
        self.alpha = alpha
        self.use_beta_sigma = use_beta_sigma
        self.tolerance = tolerance
        self.q_hat = None  # Calibrated quantile
        self._cal_scores = None  # Stored calibration scores
    
    def calibrate(
        self,
        y_cal: np.ndarray,
        mu_cal: np.ndarray,
        phi_cal: np.ndarray | None = None,
    ) -> float:
        """Calibrate on the held-out calibration set.
        
        Args:
            y_cal: True efficacy values (n_cal,).
            mu_cal: Predicted means (n_cal,).
            phi_cal: Predicted precisions (n_cal,). Required if use_beta_sigma.
        
        Returns:
            Calibrated quantile q̂.
        """
        residuals = np.abs(y_cal - mu_cal)
        
        if self.use_beta_sigma and phi_cal is not None:
            # Normalize by predicted uncertainty
            mu_safe = np.clip(mu_cal, 1e-6, 1.0 - 1e-6)
            sigma = np.sqrt(np.clip(mu_safe * (1 - mu_safe) / (1 + phi_cal), 0, None))
            sigma = np.clip(sigma, 1e-8, None)  # numerical stability
            scores = residuals / sigma
        else:
            scores = residuals
        
        n = len(scores)
        # Finite-sample adjusted quantile level
        level = np.ceil((1 - self.alpha) * (n + 1)) / n
        level = min(level, 1.0)
        
        self.q_hat = np.quantile(scores, level)
        self._cal_scores = scores
        
        return self.q_hat
    
    def predict(
        self,
        mu_test: np.ndarray,
        phi_test: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute prediction intervals for test points.
        
        Args:
            mu_test: Predicted means (n_test,).
            phi_test: Predicted precisions (n_test,).
        
        Returns:
            lower: Lower bounds of prediction intervals (n_test,).
            upper: Upper bounds of prediction intervals (n_test,).
        """
        if self.q_hat is None:
            raise RuntimeError("Must call calibrate() before predict()")
        
        if self.use_beta_sigma and phi_test is not None:
            mu_safe = np.clip(mu_test, 1e-6, 1.0 - 1e-6)
            sigma = np.sqrt(np.clip(mu_safe * (1 - mu_safe) / (1 + phi_test), 0, None))
            sigma = np.clip(sigma, 1e-8, None)
            margin = self.q_hat * sigma
        else:
            margin = self.q_hat * np.ones_like(mu_test)
        
        lower = np.clip(mu_test - margin, 0.0, 1.0)
        upper = np.clip(mu_test + margin, 0.0, 1.0)
        
        return lower, upper
    
    def evaluate_coverage(
        self,
        y_test: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> dict:
        """Evaluate coverage statistics.
        
        Returns dict with:
            - coverage: Empirical coverage (should be ≈ 1-α)
            - avg_width: Average interval width
            - median_width: Median interval width
            - coverage_by_quantile: Coverage broken down by prediction quantile
            - within_tolerance: Whether coverage is within ±tolerance of 1-α
        """
        covered = (y_test >= lower) & (y_test <= upper)
        coverage = covered.mean()
        widths = upper - lower
        
        # Coverage by prediction quantile (for group conditional coverage)
        n_groups = 5
        quantiles = np.quantile(
            (lower + upper) / 2,
            np.linspace(0, 1, n_groups + 1),
        )
        coverage_by_group = {}
        for i in range(n_groups):
            mask = (((lower + upper) / 2) >= quantiles[i]) & (((lower + upper) / 2) < quantiles[i + 1])
            if mask.sum() > 0:
                coverage_by_group[f"Q{i+1}"] = covered[mask].mean()
        
        target = 1 - self.alpha
        
        return {
            "coverage": float(coverage),
            "target_coverage": float(target),
            "avg_width": float(widths.mean()),
            "median_width": float(np.median(widths)),
            "std_width": float(widths.std()),
            "coverage_by_group": coverage_by_group,
            "within_tolerance": bool(abs(coverage - target) <= self.tolerance),
            "coverage_gap": float(coverage - target),
        }


class WeightedConformalPredictor(SplitConformalPredictor):
    """Weighted split conformal prediction for distribution shift.
    
    For Splits B (dataset-held-out) and C (cell-line-held-out),
    uses likelihood ratio weighting to account for covariate shift
    between calibration and test distributions.
    """
    
    def __init__(
        self,
        alpha: float = 0.10,
        use_beta_sigma: bool = True,
        tolerance: float = 0.02,
    ):
        super().__init__(alpha, use_beta_sigma, tolerance)
    
    def calibrate_weighted(
        self,
        y_cal: np.ndarray,
        mu_cal: np.ndarray,
        phi_cal: np.ndarray | None,
        weights: np.ndarray | None = None,
    ) -> float:
        """Calibrate with importance weights for covariate shift.
        
        Args:
            weights: Likelihood ratio weights (n_cal,).
                     If None, falls back to unweighted.
        """
        residuals = np.abs(y_cal - mu_cal)
        
        if self.use_beta_sigma and phi_cal is not None:
            sigma = np.sqrt(mu_cal * (1 - mu_cal) / (1 + phi_cal))
            sigma = np.clip(sigma, 1e-8, None)
            scores = residuals / sigma
        else:
            scores = residuals
        
        if weights is None:
            return self.calibrate(y_cal, mu_cal, phi_cal)
        
        # Weighted quantile
        weights = weights / weights.sum()
        sorted_idx = np.argsort(scores)
        sorted_scores = scores[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cum_weights = np.cumsum(sorted_weights)
        
        level = 1 - self.alpha
        idx = np.searchsorted(cum_weights, level)
        idx = min(idx, len(sorted_scores) - 1)
        
        self.q_hat = sorted_scores[idx]
        self._cal_scores = scores
        
        return self.q_hat
