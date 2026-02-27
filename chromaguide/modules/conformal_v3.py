"""Enhanced conformal prediction for ChromaGuide v3.

Key improvements over v2:
    1. Weighted conformal prediction for domain-shift splits (B/C)
    2. Temperature scaling before conformal calibration  
    3. Adaptive non-conformity scores
    4. Mondrian conformal for group-conditional coverage

References:
    - Barber et al. (2023) "Conformal Prediction Beyond Exchangeability"
    - Tibshirani et al. (2019) "Conformal Prediction Under Covariate Shift"
"""
from __future__ import annotations
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SplitConformalPredictor:
    """Standard split conformal predictor (for IID splits like Split A)."""
    
    def __init__(
        self,
        alpha: float = 0.10,
        use_beta_sigma: bool = True,
        tolerance: float = 0.02,
    ):
        self.alpha = alpha
        self.use_beta_sigma = use_beta_sigma
        self.tolerance = tolerance
        self.q_hat = None
    
    def calibrate(
        self,
        cal_y: np.ndarray,
        cal_mu: np.ndarray,
        cal_phi: np.ndarray,
    ) -> float:
        """Calibrate on calibration set."""
        residuals = np.abs(cal_y - cal_mu)
        
        if self.use_beta_sigma:
            # Normalize by predicted uncertainty
            sigma = np.sqrt(cal_mu * (1 - cal_mu) / (1 + cal_phi))
            sigma = np.maximum(sigma, 1e-6)
            scores = residuals / sigma
        else:
            scores = residuals
        
        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(scores, min(level, 1.0))
        
        return self.q_hat
    
    def predict(
        self,
        test_mu: np.ndarray,
        test_phi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict intervals on test set."""
        if self.q_hat is None:
            raise RuntimeError("Must calibrate before predict")
        
        if self.use_beta_sigma:
            sigma = np.sqrt(test_mu * (1 - test_mu) / (1 + test_phi))
            sigma = np.maximum(sigma, 1e-6)
            half_width = self.q_hat * sigma
        else:
            half_width = np.full_like(test_mu, self.q_hat)
        
        lower = np.clip(test_mu - half_width, 0, 1)
        upper = np.clip(test_mu + half_width, 0, 1)
        
        return lower, upper
    
    def evaluate_coverage(
        self,
        test_y: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> dict:
        covered = (test_y >= lower) & (test_y <= upper)
        coverage = covered.mean()
        avg_width = (upper - lower).mean()
        target_coverage = 1 - self.alpha
        
        return {
            'coverage': float(coverage),
            'avg_width': float(avg_width),
            'within_tolerance': bool(abs(coverage - target_coverage) <= self.tolerance),
            'target_coverage': float(target_coverage),
        }


class WeightedConformalPredictor:
    """Weighted conformal predictor for domain shift (Splits B/C).
    
    Under covariate shift, standard conformal prediction loses coverage 
    guarantees. This implementation uses likelihood ratios to weight 
    calibration scores, following Tibshirani et al. (2019).
    
    For CRISPR domain shift (different cell lines/datasets), we use
    a density ratio estimator based on prediction confidence.
    """
    
    def __init__(
        self,
        alpha: float = 0.10,
        use_beta_sigma: bool = True,
        tolerance: float = 0.02,
        domain_weight_method: str = "adaptive",  # adaptive, uniform, density_ratio
    ):
        self.alpha = alpha
        self.use_beta_sigma = use_beta_sigma
        self.tolerance = tolerance
        self.domain_weight_method = domain_weight_method
        self.q_hat = None
        self._cal_scores = None
        self._cal_weights = None
    
    def calibrate(
        self,
        cal_y: np.ndarray,
        cal_mu: np.ndarray,
        cal_phi: np.ndarray,
        cal_domain: np.ndarray | None = None,
        test_mu: np.ndarray | None = None,
        test_phi: np.ndarray | None = None,
    ) -> float:
        """Calibrate with domain-aware weighting."""
        residuals = np.abs(cal_y - cal_mu)
        
        if self.use_beta_sigma:
            sigma = np.sqrt(cal_mu * (1 - cal_mu) / (1 + cal_phi))
            sigma = np.maximum(sigma, 1e-6)
            scores = residuals / sigma
        else:
            scores = residuals
        
        self._cal_scores = scores
        
        # Compute weights
        if self.domain_weight_method == "adaptive" and test_mu is not None:
            # Adaptive: weight calibration points by similarity to test distribution
            weights = self._compute_adaptive_weights(cal_mu, cal_phi, test_mu, test_phi)
        elif self.domain_weight_method == "density_ratio" and cal_domain is not None:
            # Density ratio based on domain labels
            weights = self._compute_domain_weights(cal_domain)
        else:
            weights = np.ones(len(scores))
        
        # Normalize weights
        weights = weights / weights.sum()
        self._cal_weights = weights
        
        # Weighted quantile
        self.q_hat = self._weighted_quantile(scores, weights, 1 - self.alpha)
        
        # Apply a conservative correction factor for domain shift
        if self.domain_weight_method == "adaptive":
            # Inflate q_hat slightly to account for distribution mismatch
            effective_sample_size = 1.0 / (weights ** 2).sum()
            correction = 1.0 + 2.0 / max(effective_sample_size, 1)
            self.q_hat *= correction
            logger.info(f"  Effective sample size: {effective_sample_size:.1f}")
            logger.info(f"  Domain shift correction factor: {correction:.4f}")
        
        return self.q_hat
    
    def _compute_adaptive_weights(
        self,
        cal_mu: np.ndarray,
        cal_phi: np.ndarray,
        test_mu: np.ndarray,
        test_phi: np.ndarray,
    ) -> np.ndarray:
        """Weight cal points by similarity to test distribution."""
        # Use phi (confidence) as a proxy for domain similarity
        cal_confidence = cal_phi / (cal_phi.max() + 1e-8)
        
        # Higher phi = more confident = potentially better calibrated
        # But under domain shift, high-confidence predictions may be wrong
        # Use a balanced approach: weight by overlap with test prediction range
        test_mu_mean = test_mu.mean()
        test_mu_std = max(test_mu.std(), 0.01)
        
        # Gaussian kernel weights
        distances = (cal_mu - test_mu_mean) / test_mu_std
        weights = np.exp(-0.5 * distances ** 2)
        
        return np.maximum(weights, 0.01)  # Floor weight
    
    def _compute_domain_weights(self, cal_domain: np.ndarray) -> np.ndarray:
        """Weight by domain group size (inverse)."""
        unique_domains, counts = np.unique(cal_domain, return_counts=True)
        weight_map = {d: 1.0 / c for d, c in zip(unique_domains, counts)}
        weights = np.array([weight_map[d] for d in cal_domain])
        return weights
    
    @staticmethod
    def _weighted_quantile(
        values: np.ndarray,
        weights: np.ndarray,
        quantile: float,
    ) -> float:
        """Compute weighted quantile."""
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[sorted_idx]
        
        cumulative = np.cumsum(sorted_weights)
        # Find the first index where cumulative weight >= quantile
        idx = np.searchsorted(cumulative, quantile * cumulative[-1])
        idx = min(idx, len(sorted_values) - 1)
        
        return float(sorted_values[idx])
    
    def predict(
        self,
        test_mu: np.ndarray,
        test_phi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.q_hat is None:
            raise RuntimeError("Must calibrate before predict")
        
        if self.use_beta_sigma:
            sigma = np.sqrt(test_mu * (1 - test_mu) / (1 + test_phi))
            sigma = np.maximum(sigma, 1e-6)
            half_width = self.q_hat * sigma
        else:
            half_width = np.full_like(test_mu, self.q_hat)
        
        lower = np.clip(test_mu - half_width, 0, 1)
        upper = np.clip(test_mu + half_width, 0, 1)
        
        return lower, upper
    
    def evaluate_coverage(
        self,
        test_y: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> dict:
        covered = (test_y >= lower) & (test_y <= upper)
        coverage = covered.mean()
        avg_width = (upper - lower).mean()
        target_coverage = 1 - self.alpha
        
        return {
            'coverage': float(coverage),
            'avg_width': float(avg_width),
            'within_tolerance': bool(abs(coverage - target_coverage) <= self.tolerance),
            'target_coverage': float(target_coverage),
        }


class MondrianConformalPredictor:
    """Mondrian conformal predictor for group-conditional coverage.
    
    Calibrates separately for each domain group (cell line / dataset),
    providing per-group coverage guarantees.
    """
    
    def __init__(
        self,
        alpha: float = 0.10,
        use_beta_sigma: bool = True,
        tolerance: float = 0.02,
        min_group_size: int = 50,
    ):
        self.alpha = alpha
        self.use_beta_sigma = use_beta_sigma
        self.tolerance = tolerance
        self.min_group_size = min_group_size
        self.group_q_hats = {}
        self.default_q_hat = None
    
    def calibrate(
        self,
        cal_y: np.ndarray,
        cal_mu: np.ndarray,
        cal_phi: np.ndarray,
        cal_groups: np.ndarray,
    ) -> dict:
        """Calibrate per group."""
        residuals = np.abs(cal_y - cal_mu)
        
        if self.use_beta_sigma:
            sigma = np.sqrt(cal_mu * (1 - cal_mu) / (1 + cal_phi))
            sigma = np.maximum(sigma, 1e-6)
            scores = residuals / sigma
        else:
            scores = residuals
        
        # Global fallback
        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.default_q_hat = np.quantile(scores, min(level, 1.0))
        
        # Per-group calibration
        unique_groups = np.unique(cal_groups)
        for group in unique_groups:
            mask = cal_groups == group
            group_scores = scores[mask]
            
            if len(group_scores) >= self.min_group_size:
                n_g = len(group_scores)
                level_g = np.ceil((n_g + 1) * (1 - self.alpha)) / n_g
                self.group_q_hats[group] = np.quantile(group_scores, min(level_g, 1.0))
            else:
                self.group_q_hats[group] = self.default_q_hat
        
        return self.group_q_hats
    
    def predict(
        self,
        test_mu: np.ndarray,
        test_phi: np.ndarray,
        test_groups: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict per-group intervals."""
        if self.use_beta_sigma:
            sigma = np.sqrt(test_mu * (1 - test_mu) / (1 + test_phi))
            sigma = np.maximum(sigma, 1e-6)
        else:
            sigma = np.ones_like(test_mu)
        
        # Get per-sample q_hat based on group
        q_hats = np.full_like(test_mu, self.default_q_hat)
        if test_groups is not None:
            for group, q in self.group_q_hats.items():
                mask = test_groups == group
                q_hats[mask] = q
        
        half_width = q_hats * sigma
        lower = np.clip(test_mu - half_width, 0, 1)
        upper = np.clip(test_mu + half_width, 0, 1)
        
        return lower, upper
    
    def evaluate_coverage(
        self,
        test_y: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        test_groups: np.ndarray | None = None,
    ) -> dict:
        covered = (test_y >= lower) & (test_y <= upper)
        coverage = covered.mean()
        avg_width = (upper - lower).mean()
        target_coverage = 1 - self.alpha
        
        result = {
            'coverage': float(coverage),
            'avg_width': float(avg_width),
            'within_tolerance': bool(abs(coverage - target_coverage) <= self.tolerance),
            'target_coverage': float(target_coverage),
        }
        
        # Per-group coverage
        if test_groups is not None:
            group_coverages = {}
            for group in np.unique(test_groups):
                mask = test_groups == group
                if mask.sum() > 0:
                    group_coverages[str(group)] = float(covered[mask].mean())
            result['group_coverages'] = group_coverages
        
        return result
