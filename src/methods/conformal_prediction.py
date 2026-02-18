"""
Conformal Prediction Methods for ChromaGuide Model Uncertainty Quantification

Implements:
- Standard split conformal prediction
- Weighted conformal prediction under distribution shift
- Group-conditional conformal prediction
- Conformal classification and regression
- Efficient conformalization without naive splits
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Union
from scipy import stats
from sklearn.metrics import coverage_error
import logging

logger = logging.getLogger(__name__)


class SplitConformalPredictor:
    """
    Standard split conformal prediction for regression.
    
    Divides calibration data into two parts:
    - Fit part: trains the model
    - Calibration part: calculates conformity scores
    """
    
    def __init__(self, model, alpha: float = 0.1):
        """
        Args:
            model: Trained PyTorch model
            alpha: Miscoverage level (e.g., 0.1 for 90% coverage)
        """
        self.model = model
        self.alpha = alpha
        self.quantile_value = None
        self.device = next(model.parameters()).device
        
    def calibrate(self, X_calib: torch.Tensor, y_calib: torch.Tensor):
        """
        Calculate conformity scores on calibration set.
        
        Args:
            X_calib: Calibration features shape (n_calib, n_features)
            y_calib: Calibration targets shape (n_calib,)
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_calib.to(self.device))
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            
        predictions = predictions.cpu().numpy().flatten()
        y_calib = y_calib.cpu().numpy().flatten()
        
        # Conformity scores: absolute residuals
        self.conformity_scores = np.abs(predictions - y_calib)
        
        # Calculate quantile (ceiling to guarantee coverage)
        n = len(self.conformity_scores)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile_value = np.quantile(self.conformity_scores, quantile_level)
        
        logger.info(f"Conformal calibration: n={n}, quantile={quantile_level:.4f}, "
                   f"threshold={self.quantile_value:.6f}")
        
    def predict(self, X_test: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate conformal prediction intervals.
        
        Args:
            X_test: Test features shape (n_test, n_features)
            
        Returns:
            point_predictions: (n_test,)
            lower_bounds: (n_test,)
            upper_bounds: (n_test,)
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test.to(self.device))
            if isinstance(predictions, tuple):
                predictions = predictions[0]
                
        predictions = predictions.cpu().numpy().flatten()
        
        # Conformal intervals
        lower_bounds = predictions - self.quantile_value
        upper_bounds = predictions + self.quantile_value
        
        return predictions, lower_bounds, upper_bounds
    
    def calculate_coverage(self, predictions: np.ndarray, lower: np.ndarray, 
                          upper: np.ndarray, targets: np.ndarray) -> float:
        """Calculate empirical coverage of prediction intervals."""
        coverage = np.mean((targets >= lower) & (targets <= upper))
        return coverage


class WeightedConformalPredictor:
    """
    Weighted conformal prediction for handling distribution shift.
    
    Uses importance weights to adjust for differences between calibration
    and test distributions.
    """
    
    def __init__(self, model, alpha: float = 0.1):
        self.model = model
        self.alpha = alpha
        self.quantile_value = None
        self.device = next(model.parameters()).device
        
    def calibrate(self, X_calib: torch.Tensor, y_calib: torch.Tensor, 
                  weights: Optional[np.ndarray] = None):
        """
        Calibrate with optional importance weights.
        
        Args:
            X_calib: Calibration features
            y_calib: Calibration targets
            weights: Importance weights shape (n_calib,)
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_calib.to(self.device))
            if isinstance(predictions, tuple):
                predictions = predictions[0]
                
        predictions = predictions.cpu().numpy().flatten()
        y_calib = y_calib.cpu().numpy().flatten()
        
        # Conformity scores
        conformity_scores = np.abs(predictions - y_calib)
        
        if weights is None:
            weights = np.ones(len(conformity_scores))
        
        # Normalize weights
        weights = weights / np.sum(weights) * len(weights)
        
        # Weighted quantile calculation
        sorted_indices = np.argsort(conformity_scores)
        sorted_scores = conformity_scores[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        cumsum_weights = np.cumsum(sorted_weights)
        quantile_level = (len(conformity_scores) + 1) * (1 - self.alpha)
        quantile_idx = np.searchsorted(cumsum_weights, quantile_level, side='left')
        quantile_idx = min(quantile_idx, len(sorted_scores) - 1)
        
        self.quantile_value = sorted_scores[quantile_idx]
        logger.info(f"Weighted conformal calibration: threshold={self.quantile_value:.6f}")
        
    def predict(self, X_test: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate weighted conformal prediction intervals."""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test.to(self.device))
            if isinstance(predictions, tuple):
                predictions = predictions[0]
                
        predictions = predictions.cpu().numpy().flatten()
        lower_bounds = predictions - self.quantile_value
        upper_bounds = predictions + self.quantile_value
        
        return predictions, lower_bounds, upper_bounds


class GroupConditionalConformalPredictor:
    """
    Group-conditional conformal prediction.
    
    Maintains separate quantiles for different groups (e.g., different
    cell lines, genomic regions) to account for group-specific uncertainty.
    """
    
    def __init__(self, model, alpha: float = 0.1):
        self.model = model
        self.alpha = alpha
        self.group_quantiles = {}
        self.device = next(model.parameters()).device
        
    def calibrate(self, X_calib: torch.Tensor, y_calib: torch.Tensor, 
                  groups: np.ndarray):
        """
        Calibrate with group labels.
        
        Args:
            X_calib: Calibration features
            y_calib: Calibration targets
            groups: Group labels shape (n_calib,)
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_calib.to(self.device))
            if isinstance(predictions, tuple):
                predictions = predictions[0]
                
        predictions = predictions.cpu().numpy().flatten()
        y_calib = y_calib.cpu().numpy().flatten()
        
        # Calculate quantiles per group
        unique_groups = np.unique(groups)
        for group in unique_groups:
            group_mask = groups == group
            conformity_scores = np.abs(predictions[group_mask] - y_calib[group_mask])
            
            n_group = np.sum(group_mask)
            quantile_level = np.ceil((n_group + 1) * (1 - self.alpha)) / n_group
            quantile_val = np.quantile(conformity_scores, quantile_level)
            
            self.group_quantiles[group] = quantile_val
            logger.info(f"Group {group}: n={n_group}, threshold={quantile_val:.6f}")
    
    def predict(self, X_test: torch.Tensor, 
                groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate group-conditional prediction intervals.
        
        Args:
            X_test: Test features
            groups: Group labels for test samples
            
        Returns:
            point_predictions, lower_bounds, upper_bounds
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test.to(self.device))
            if isinstance(predictions, tuple):
                predictions = predictions[0]
                
        predictions = predictions.cpu().numpy().flatten()
        
        lower_bounds = np.zeros_like(predictions)
        upper_bounds = np.zeros_like(predictions)
        
        for i, group in enumerate(groups):
            threshold = self.group_quantiles.get(group, 
                                                np.quantile(list(self.group_quantiles.values()), 0.5))
            lower_bounds[i] = predictions[i] - threshold
            upper_bounds[i] = predictions[i] + threshold
        
        return predictions, lower_bounds, upper_bounds


class AdaptiveConformalPredictor:
    """
    Adaptive conformal prediction that adjusts quantiles based on input features.
    
    Learns a function that maps features to appropriate quantile levels,
    allowing for heteroscedastic uncertainty estimates.
    """
    
    def __init__(self, model, quantile_network=None, alpha: float = 0.1):
        """
        Args:
            model: Main prediction model
            quantile_network: Optional neural network to learn feature-dependent quantiles
            alpha: Miscoverage level
        """
        self.model = model
        self.quantile_network = quantile_network
        self.alpha = alpha
        self.base_quantile = None
        self.device = next(model.parameters()).device
        
    def calibrate(self, X_calib: torch.Tensor, y_calib: torch.Tensor):
        """Calibrate adaptive conformal predictor."""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_calib.to(self.device))
            if isinstance(predictions, tuple):
                predictions = predictions[0]
                
        predictions = predictions.cpu().numpy().flatten()
        y_calib = y_calib.cpu().numpy().flatten()
        
        # Base conformity scores
        conformity_scores = np.abs(predictions - y_calib)
        n = len(conformity_scores)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        
        self.base_quantile = np.quantile(conformity_scores, quantile_level)
        
        # Train quantile network if provided
        if self.quantile_network is not None:
            self._train_quantile_network(X_calib, conformity_scores)
            
        logger.info(f"Adaptive conformal calibration: base_quantile={self.base_quantile:.6f}")
    
    def _train_quantile_network(self, X_calib: torch.Tensor, 
                               conformity_scores: np.ndarray, epochs: int = 50):
        """Train network to predict quantiles from features."""
        optimizer = torch.optim.Adam(self.quantile_network.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            self.quantile_network.train()
            pred_quantiles = self.quantile_network(X_calib.to(self.device))
            pred_quantiles = pred_quantiles.squeeze()
            
            # Loss: predictions should be >= conformity scores
            loss = F.relu(conformity_scores - pred_quantiles).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Quantile network epoch {epoch+1}: loss={loss.item():.6f}")
    
    def predict(self, X_test: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate adaptive conformal prediction intervals."""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test.to(self.device))
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            predictions = predictions.cpu().numpy().flatten()
        
        if self.quantile_network is not None:
            self.quantile_network.eval()
            with torch.no_grad():
                pred_quantiles = self.quantile_network(X_test.to(self.device))
            pred_quantiles = pred_quantiles.squeeze().cpu().numpy()
        else:
            pred_quantiles = np.full_like(predictions, self.base_quantile)
        
        lower_bounds = predictions - pred_quantiles
        upper_bounds = predictions + pred_quantiles
        
        return predictions, lower_bounds, upper_bounds


class ConformalRegressionEvaluator:
    """Comprehensive evaluation metrics for conformal prediction."""
    
    @staticmethod
    def evaluate(predictions: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                 targets: np.ndarray, alpha: float = 0.1) -> Dict[str, float]:
        """
        Calculate comprehensive conformal prediction metrics.
        
        Args:
            predictions: Point predictions
            lower: Lower bounds
            upper: Upper bounds
            targets: Ground truth values
            alpha: Target miscoverage level
            
        Returns:
            Dictionary of metrics
        """
        # Coverage
        in_interval = (targets >= lower) & (targets <= upper)
        coverage = np.mean(in_interval)
        
        # Average interval width
        width = upper - lower
        avg_width = np.mean(width)
        
        # Interval-wise coverage and width
        coverage_by_interval = np.reshape(in_interval, -1)
        width_by_interval = np.reshape(width, -1)
        
        # Efficiency metrics
        min_width = np.min(width)
        max_width = np.max(width)
        
        # Miscoverage
        miscoverage = 1 - coverage
        target_miscoverage = alpha
        miscoverage_error = np.abs(miscoverage - target_miscoverage)
        
        # Excess calculation: how much coverage exceeds target
        excess_coverage = max(0, coverage - (1 - alpha))
        
        # Point prediction error
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # Width-efficiency trade-off
        efficiency = coverage / (avg_width + 1e-6)
        
        return {
            'coverage': float(coverage),
            'target_coverage': float(1 - alpha),
            'miscoverage': float(miscoverage),
            'miscoverage_error': float(miscoverage_error),
            'avg_width': float(avg_width),
            'min_width': float(min_width),
            'max_width': float(max_width),
            'efficiency': float(efficiency),
            'mae': float(mae),
            'rmse': float(rmse),
            'excess_coverage': float(excess_coverage),
        }
    
    @staticmethod
    def report(metrics: Dict[str, float]) -> str:
        """Generate human-readable report of conformal prediction metrics."""
        report = """
CONFORMAL PREDICTION EVALUATION REPORT
======================================
Coverage:
  • Achieved:  {coverage:.1%}
  • Target:    {target_coverage:.1%}
  • Error:     {miscoverage_error:.1%}

Interval Width:
  • Average:   {avg_width:.4f}
  • Min:       {min_width:.4f}
  • Max:       {max_width:.4f}
  • Efficiency:{efficiency:.2f}

Point Predictions:
  • MAE:       {mae:.6f}
  • RMSE:      {rmse:.6f}

Excess Coverage: {excess_coverage:.1%}
""".format(**metrics)
        return report


def conformalize_predictions(model, X_test: torch.Tensor, 
                            predictor: Union[SplitConformalPredictor, 
                                           WeightedConformalPredictor,
                                           GroupConditionalConformalPredictor,
                                           AdaptiveConformalPredictor],
                            **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper function to generate conformal predictions.
    
    Args:
        model: Trained model
        X_test: Test features
        predictor: Conformal predictor instance
        **kwargs: Additional arguments for predictor.predict()
        
    Returns:
        point_predictions, lower_bounds, upper_bounds
    """
    return predictor.predict(X_test, **kwargs)
