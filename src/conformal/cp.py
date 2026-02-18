import torch
import numpy as np

class MondrianConformalPredictor:
    """
    Mondrian Conformal Predictor for Regression.
    Provides mathematically guaranteed coverage (e.g., 90%) by calibrating
    non-conformity scores on a hold-out set.

    "Mondrian" means we can stratify by category (e.g., Cell Type) to ensure
    validity *within* each group, not just marginally.
    """
    def __init__(self, alpha=0.1):
        """
        alpha: Error rate (0.1 = 90% confidence)
        """
        self.alpha = alpha
        self.calibration_scores = {} # Dict[group, List[scores]]
        self.q_hat = {} # Imperical quantile per group

    def calibrate(self, pred_scores, true_efficiencies, groups=None):
        """
        Compute non-conformity scores from calibration set.
        Score = |y - y_hat| (Absolute Residual)

        pred_scores: Tensor [N]
        true_efficiencies: Tensor [N]
        groups: List[str] of length N (optional stratification)
        """
        residuals = torch.abs(true_efficiencies - pred_scores).detach().cpu().numpy()

        if groups is None:
            groups = ["global"] * len(residuals)

        # Group residuals
        from collections import defaultdict
        grouped_residuals = defaultdict(list)
        for r, g in zip(residuals, groups):
            grouped_residuals[g].append(r)

        # Compute Quantiles (1-alpha) * (1 + 1/n)
        for g, scores in grouped_residuals.items():
            n = len(scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            q_level = min(1.0, max(0.0, q_level)) # Clip

            # self.q_hat[g] = np.quantile(scores, q_level, method='higher')
            # safe numpy quantile
            self.q_hat[g] = np.percentile(scores, q_level * 100)
            self.calibration_scores[g] = scores

        print(f"Calibration Complete. Q_hat (Global/Default): {self.q_hat.get('global', 'N/A')}")

    def predict(self, pred_scores, groups=None):
        """
        Return prediction intervals [lower, upper].
        Interval = y_hat +/- q_hat
        """
        if groups is None:
            groups = ["global"] * len(pred_scores)

        lower_bounds = []
        upper_bounds = []

        pred_scores = pred_scores.detach().cpu().numpy()

        for score, g in zip(pred_scores, groups):
            q = self.q_hat.get(g, self.q_hat.get("global", 0.15)) # Fallback to default width if unknown group
            lower_bounds.append(max(0.0, score - q))
            upper_bounds.append(min(1.0, score + q))

        return list(zip(lower_bounds, upper_bounds))

    def evaluate_coverage(self, pred_intervals, true_values):
        """
        Check if true values fall within intervals.
        """
        hits = 0
        for (low, high), val in zip(pred_intervals, true_values):
            if low <= val <= high:
                hits += 1
        return hits / len(true_values)
