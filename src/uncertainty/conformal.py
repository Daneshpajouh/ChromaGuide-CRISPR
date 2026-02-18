import numpy as np
import torch

class ConformalPredictor:
    """
    Split Conformal Prediction for DeepMEns (Regression).
    Guarantees that the true efficiency is within [lower, upper] with probability 1-alpha.
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q = None # Calibrated quantile score
        self.residuals = []

    def calibrate(self, y_true, y_pred):
        """
        Compute residuals on a calibration set (held-out from training).
        y_true, y_pred: Arrays or Tensors.
        """
        if isinstance(y_true, torch.Tensor): y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().numpy()

        # Absolute residuals for regression
        scores = np.abs(y_true - y_pred)
        self.residuals = scores

        # Compute (1-alpha) quantile
        # n = len(scores)
        # q = (1 - alpha) * (1 + 1/n) # Finite sample correction
        n = len(scores)
        q_val = np.quantile(scores, np.ceil((n+1) * (1 - self.alpha)) / n, method='higher')
        self.q = q_val
        print(f"Calibrated Conformal Interval (alpha={self.alpha}): +/- {self.q:.4f}")

    def predict(self, y_pred):
        """
        Return prediction intervals [lower, upper]
        """
        if self.q is None:
            raise ValueError("Predictor not calibrated!")

        if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().numpy()

        lower = y_pred - self.q
        upper = y_pred + self.q

        # Clip to [0,1] for Efficiency
        lower = np.clip(lower, 0, 1)
        upper = np.clip(upper, 0, 1)

        return lower, upper

if __name__ == "__main__":
    # Test
    y_cal_true = np.random.rand(100)
    y_cal_pred = y_cal_true + np.random.normal(0, 0.05, 100)

    cp = ConformalPredictor(alpha=0.1)
    cp.calibrate(y_cal_true, y_cal_pred)

    y_test = np.array([0.5, 0.8])
    lo, hi = cp.predict(y_test)
    print(f"Pred: {y_test}")
    print(f"Intervals: [{lo}, {hi}]")
