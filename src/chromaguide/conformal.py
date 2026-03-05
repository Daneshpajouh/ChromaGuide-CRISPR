"""Conformal Prediction for ChromaGuide using Beta Distribution.

Implements distribution-free uncertainty quantification for CRISPR
on-target efficacy prediction.

Methodology from Section 5.3 of the ChromaGuide PhD Proposal:
  - Uses split conformal prediction.
  - Conformity scores are derived from the Beta distribution quantile function.
  - Provides mathematically guaranteed coverage 1 - alpha.
"""
import numpy as np
from scipy.stats import beta
from pathlib import Path
from typing import Tuple


class BetaConformalPredictor:
    """Split Conformal Predictor for Beta Regression on-target models.

    The conformity score is the maximum of the distances from the
    true efficacy to the lower and upper quantiles of the predicted
    Beta distribution.
    """
    def __init__(self, alpha: float = 0.1):
        """Initialize with desired miscoverage rate.

        Args:
            alpha: Miscoverage rate (e.g., 0.1 for 90% coverage).
        """
        self.alpha = alpha
        self.q = None  # (1-alpha) quantile of conformity scores

    def compute_conformity_scores(
        self,
        mu: np.ndarray,
        phi: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Compute conformity scores for a calibrated set.

        s_i = max(F^{-1}(alpha/2) - y_i, y_i - F^{-1}(1 - alpha/2))

        Args:
            mu: Mean estimates (n_samples,)
            phi: Concentration estimates (n_samples,)
            y: True efficacy values (n_samples,)

        Returns:
            Conformity scores (n_samples,)
        """
        # alpha_param = mu * phi
        # beta_param = (1 - mu) * phi
        alpha_p = mu * phi
        beta_p = (1 - mu) * phi

        lower_q = beta.ppf(self.alpha / 2, alpha_p, beta_p)
        upper_q = beta.ppf(1 - self.alpha / 2, alpha_p, beta_p)

        # Nonconformity should be non-negative:
        # 0 for points inside [lower_q, upper_q], positive distance outside.
        scores = np.maximum(np.maximum(lower_q - y, y - upper_q), 0.0)
        return scores

    def calibrate(
        self,
        mu: np.ndarray,
        phi: np.ndarray,
        y: np.ndarray,
    ):
        """Fit the predictor using a calibration set.

        Args:
            mu, phi: Model outputs for calibration set.
            y: True values for calibration set.
        """
        scores = self.compute_conformity_scores(mu, phi, y)
        n = len(scores)
        # Empirical quantile: (n+1)(1-alpha)/n
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = np.clip(level, 0, 1)
        self.q = float(np.quantile(scores, level, method='higher'))

    def predict_intervals(
        self,
        mu: np.ndarray,
        phi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Produce prediction intervals for new test points.

        Args:
            mu, phi: Model outputs for test set.

        Returns:
            (lower_bounds, upper_bounds)
        """
        if self.q is None:
            raise ValueError("Predictor must be calibrated before use.")

        alpha_p = mu * phi
        beta_p = (1 - mu) * phi

        # Initial interval from distribution quantiles
        lower_q = beta.ppf(self.alpha / 2, alpha_p, beta_p)
        upper_q = beta.ppf(1 - self.alpha / 2, alpha_p, beta_p)

        # Expand by the calibrated quantile q
        # [lower - q, upper + q] and clip to [0, 1]
        q = max(float(self.q), 0.0)
        lower_bounds = np.clip(lower_q - q, 1e-6, 1.0 - 1e-6)
        upper_bounds = np.clip(upper_q + q, 1e-6, 1.0 - 1e-6)

        return lower_bounds, upper_bounds

    def get_interval_width(
        self,
        mu: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        """Compute the width of the prediction intervals (sigma_CI)."""
        lower, upper = self.predict_intervals(mu, phi)
        return upper - lower

    def save_calibration(self, path: str) -> None:
        """Save calibrated conformal state (alpha and q)."""
        if self.q is None:
            raise ValueError("Cannot save calibration before calling calibrate().")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, alpha=float(self.alpha), q=float(self.q))

    def load_calibration(self, path: str) -> None:
        """Load calibrated conformal state (alpha and q)."""
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {in_path}")

        # Backward-compatible support:
        # - npz: keys {'q', optional 'alpha'}
        # - npy: single scalar/array with q
        if in_path.suffix == ".npy":
            q_arr = np.load(in_path, allow_pickle=False)
            self.q = float(np.asarray(q_arr).reshape(-1)[0])
            return

        data = np.load(in_path, allow_pickle=False)
        if "q" not in data:
            raise ValueError(f"Invalid calibration file (missing q): {in_path}")

        if "alpha" in data:
            self.alpha = float(data["alpha"])
        self.q = float(data["q"])
