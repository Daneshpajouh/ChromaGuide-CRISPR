"""Prediction head module for ChromaGuide.

Implements Beta regression for bounded on-target efficacy prediction.
The Beta distribution naturally models outcomes in (0,1) and provides
calibrated uncertainty through its concentration parameters.

Also includes split conformal prediction for distribution-free
prediction intervals.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class BetaRegressionHead(nn.Module):
    """Bounded-outcome prediction head using Beta regression.

    Predicts parameters (mu, phi) of a Beta distribution:
      alpha = mu * phi
      beta = (1 - mu) * phi
    where mu is the mean prediction and phi is the concentration.

    The loss is the negative log-likelihood of the Beta distribution.
    """
    def __init__(
        self,
        d_model: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Mean prediction (sigmoid output for [0,1])
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Concentration (softplus for positive output)
        self.phi_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict Beta distribution parameters.

        Args:
            z: Fused embedding (batch, d_model)

        Returns:
            Dict with keys:
              'mu': Mean prediction (batch, 1) in (0, 1)
              'phi': Concentration (batch, 1) > 0
              'alpha': Beta alpha parameter (batch, 1)
              'beta': Beta beta parameter (batch, 1)
        """
        h = self.shared(z)

        mu = self.mu_head(h).clamp(1e-4, 1 - 1e-4)  # avoid boundary
        phi = self.phi_head(h) + 2.0  # minimum concentration of 2

        alpha = mu * phi
        beta = (1 - mu) * phi

        return {
            'mu': mu,
            'phi': phi,
            'alpha': alpha,
            'beta': beta,
        }


class BetaNLL(nn.Module):
    """Negative log-likelihood loss for Beta regression.

    L = -log Beta(y | alpha, beta)
      = -[(alpha-1)*log(y) + (beta-1)*log(1-y) - log B(alpha, beta)]
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Beta NLL loss.

        Args:
            alpha: Beta alpha parameter (batch, 1)
            beta: Beta beta parameter (batch, 1)
            y: True efficacy values in (0, 1) (batch, 1)

        Returns:
            Scalar loss
        """
        y = y.clamp(self.eps, 1 - self.eps)

        # Log-beta function via lgamma
        log_beta_fn = (
            torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
        )

        # Negative log-likelihood
        nll = (
            log_beta_fn
            - (alpha - 1) * torch.log(y)
            - (beta - 1) * torch.log(1 - y)
        )

        return nll.mean()


class SplitConformalPredictor:
    """Split conformal prediction for calibrated prediction intervals.

    Uses the exchangeability assumption: given calibration scores from
    a held-out calibration set, produces distribution-free prediction
    intervals at any desired coverage level.

    For dataset/cell-line shift, uses weighted conformal calibration
    with estimated importance weights / likelihood ratios.
    """
    def __init__(self, alpha: float = 0.10):
        """Initialize with desired miscoverage rate alpha.

        Args:
            alpha: Target miscoverage rate (e.g., 0.10 for 90% coverage)
        """
        self.alpha = alpha
        self.calibration_scores = None
        self.quantile = None

    def calibrate(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ):
        """Calibrate using held-out data.

        Args:
            predictions: Model predictions on calibration set
            true_values: True values for calibration set
            weights: Optional importance weights for weighted conformal
        """
        # Conformity scores: absolute residuals
        self.calibration_scores = np.abs(predictions - true_values)
        n = len(self.calibration_scores)

        if weights is not None:
            # Weighted conformal for distribution shift
            sorted_idx = np.argsort(self.calibration_scores)
            sorted_scores = self.calibration_scores[sorted_idx]
            sorted_weights = weights[sorted_idx]
            cumulative_weights = np.cumsum(sorted_weights)
            cumulative_weights /= cumulative_weights[-1]

            target = 1 - self.alpha
            idx = np.searchsorted(cumulative_weights, target)
            self.quantile = sorted_scores[min(idx, n - 1)]
        else:
            # Standard split conformal
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            self.quantile = np.quantile(
                self.calibration_scores,
                min(q_level, 1.0),
            )

    def predict(
        self, point_predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Produce prediction intervals.

        Args:
            point_predictions: New point predictions

        Returns:
            (lower_bounds, upper_bounds) prediction intervals
        """
        if self.quantile is None:
            raise RuntimeError('Must call calibrate() before predict()')

        lower = np.clip(point_predictions - self.quantile, 0.0, 1.0)
        upper = np.clip(point_predictions + self.quantile, 0.0, 1.0)

        return lower, upper

    def evaluate_coverage(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        true_values: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate prediction interval quality."""
        covered = (true_values >= lower) & (true_values <= upper)
        coverage = covered.mean()
        avg_width = (upper - lower).mean()

        return {
            'coverage': float(coverage),
            'avg_width': float(avg_width),
            'target_coverage': 1 - self.alpha,
        }
