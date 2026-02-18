#!/usr/bin/env python3
"""
Comprehensive Model Validation
==============================

Advanced validation checks and testing for production deployment:
- Cross-validation strategies
- Out-of-distribution detection
- Fairness and bias checking
- Robustness testing
- Reproducibility verification
- Model versioning and registry

Usage:
    validator = ModelValidator(model)
    results = validator.comprehensive_validation(X_test, y_test)
    validator.report_validation_results(results)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import json
import logging
from datetime import datetime
from scipy.stats import ks_2samp, entropy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    score: float
    message: str
    timestamp: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'test': self.test_name,
            'passed': self.passed,
            'score': self.score,
            'message': self.message,
            'timestamp': self.timestamp
        }


class MetricsValidator:
    """Validate model metrics and performance."""
    
    @staticmethod
    def check_metric_bounds(metric_name: str, value: float, 
                          bounds: Tuple[float, float]) -> ValidationResult:
        """Check if metric is within expected bounds."""
        lower, upper = bounds
        passed = lower <= value <= upper
        
        return ValidationResult(
            test_name=f"metric_{metric_name}_bounds",
            passed=passed,
            score=value,
            message=f"{metric_name}={value:.4f} within [{lower}, {upper}]",
            timestamp=datetime.now().isoformat(),
            metadata={'lower': lower, 'upper': upper}
        )
    
    @staticmethod
    def check_metric_consistency(metrics: Dict[str, float], 
                                baseline: Dict[str, float],
                                tolerance: float = 0.05) -> ValidationResult:
        """Check metric consistency with baseline."""
        differences = {}
        
        for key in baseline:
            if key in metrics:
                diff = abs(metrics[key] - baseline[key]) / (abs(baseline[key]) + 1e-6)
                differences[key] = diff
        
        all_within = all(d <= tolerance for d in differences.values())
        avg_diff = np.mean(list(differences.values())) if differences else 0
        
        return ValidationResult(
            test_name="metric_consistency",
            passed=all_within,
            score=avg_diff,
            message=f"Metrics consistent with baseline (avg diff: {avg_diff:.4f})",
            timestamp=datetime.now().isoformat(),
            metadata={'differences': differences, 'tolerance': tolerance}
        )


class DataValidator:
    """Validate input data quality."""
    
    @staticmethod
    def check_data_shape(X: np.ndarray, y: np.ndarray,
                        expected_features: Optional[int] = None) -> ValidationResult:
        """Check data dimensions."""
        n_samples, n_features = X.shape
        
        issues = []
        if n_samples < 10:
            issues.append(f"Too few samples: {n_samples}")
        if n_features < 2 and expected_features != n_features:
            issues.append(f"Too few features: {n_features}")
        if len(y) != n_samples:
            issues.append(f"Label mismatch: {len(y)} labels vs {n_samples} samples")
        
        passed = len(issues) == 0
        message = "; ".join(issues) if issues else "Data shape valid"
        
        return ValidationResult(
            test_name="data_shape",
            passed=passed,
            score=float(n_samples * n_features),
            message=message,
            timestamp=datetime.now().isoformat(),
            metadata={'n_samples': n_samples, 'n_features': n_features}
        )
    
    @staticmethod
    def check_data_quality(X: np.ndarray) -> ValidationResult:
        """Check data quality (missing values, duplicates, etc.)."""
        issues = []
        
        # Check for NaN
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            issues.append(f"{nan_count} NaN values")
        
        # Check for inf
        inf_count = np.isinf(X).sum()
        if inf_count > 0:
            issues.append(f"{inf_count} infinite values")
        
        # Check for zero variance features
        zero_var_features = np.where(X.std(axis=0) < 1e-10)[0]
        if len(zero_var_features) > 0:
            issues.append(f"{len(zero_var_features)} zero-variance features")
        
        passed = len(issues) == 0
        message = "; ".join(issues) if issues else "Data quality good"
        
        return ValidationResult(
            test_name="data_quality",
            passed=passed,
            score=float(nan_count + inf_count),
            message=message,
            timestamp=datetime.now().isoformat(),
            metadata={'nan_count': int(nan_count), 'inf_count': int(inf_count)}
        )


class ModelValidator:
    """Main model validation coordinator."""
    
    def __init__(self, model):
        self.model = model
        self.results: List[ValidationResult] = []
    
    def validate_predictions(self, y_pred: np.ndarray, y_true: np.ndarray) -> ValidationResult:
        """Validate predictions are reasonable."""
        from sklearn.metrics import mean_squared_error, r2_score
        from scipy.stats import spearmanr, pearsonr
        
        # Check prediction range
        if np.any(y_pred < 0) or np.any(y_pred > 1):
            logger.warning("Predictions outside [0,1] range")
        
        # Compute metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        spearman_r, _ = spearmanr(y_true, y_pred)
        
        # Check quality
        passed = rmse < 0.5 and r2 > 0.3 and spearman_r > 0.3
        
        return ValidationResult(
            test_name="prediction_quality",
            passed=passed,
            score=r2,
            message=f"RMSE={rmse:.4f}, R²={r2:.4f}, Spearman={spearman_r:.4f}",
            timestamp=datetime.now().isoformat(),
            metadata={'rmse': float(rmse), 'r2': float(r2), 'spearman': float(spearman_r)}
        )
    
    def check_reproducibility(self, X: np.ndarray, n_runs: int = 3) -> ValidationResult:
        """Check if predictions are reproducible."""
        predictions = []
        
        for _ in range(n_runs):
            pred = self.model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Check consistency
        consistency = 1 - np.std(predictions, axis=0).mean() / (predictions.mean() + 1e-6)
        consistency = max(0, min(1, consistency))
        
        # All predictions should be nearly identical
        passed = consistency > 0.99
        
        return ValidationResult(
            test_name="reproducibility",
            passed=passed,
            score=consistency,
            message=f"Prediction consistency: {consistency:.4f}",
            timestamp=datetime.now().isoformat(),
            metadata={'n_runs': n_runs, 'consistency': float(consistency)}
        )
    
    def check_out_of_distribution(self, X_train: np.ndarray, 
                                 X_test: np.ndarray) -> ValidationResult:
        """Detect out-of-distribution test samples."""
        # Simple KS test on feature distributions
        ks_stats = []
        
        for feature_idx in range(X_train.shape[1]):
            stat, _ = ks_2samp(X_train[:, feature_idx], X_test[:, feature_idx])
            ks_stats.append(stat)
        
        mean_ks = np.mean(ks_stats)
        
        # Flag if distributions are significantly different
        passed = mean_ks < 0.3  # Threshold for similarity
        
        return ValidationResult(
            test_name="ood_detection",
            passed=passed,
            score=mean_ks,
            message=f"Distribution shift score: {mean_ks:.4f}",
            timestamp=datetime.now().isoformat(),
            metadata={'ks_statistics': [float(s) for s in ks_stats]}
        )
    
    def check_fairness(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray,
                      protected_feature_idx: int = 0) -> ValidationResult:
        """Check for bias across protected groups."""
        from sklearn.metrics import accuracy_score
        
        # Split by protected feature
        feature_values = np.unique(X[:, protected_feature_idx])
        
        accuracies = []
        for val in feature_values:
            mask = X[:, protected_feature_idx] == val
            if mask.sum() > 0:
                acc = accuracy_score(y[mask], (y_pred[mask] > 0.5).astype(int))
                accuracies.append(acc)
        
        accuracy_spread = max(accuracies) - min(accuracies) if accuracies else 0
        
        # Threshold for fairness
        passed = accuracy_spread < 0.1
        
        return ValidationResult(
            test_name="fairness",
            passed=passed,
            score=1 - accuracy_spread,
            message=f"Accuracy spread across groups: {accuracy_spread:.4f}",
            timestamp=datetime.now().isoformat(),
            metadata={'accuracies': [float(a) for a in accuracies]}
        )
    
    def check_robustness(self, X: np.ndarray, noise_std: float = 0.1,
                        n_trials: int = 10) -> ValidationResult:
        """Check robustness to input noise."""
        predictions_clean = self.model.predict(X)
        
        prediction_errors = []
        
        for _ in range(n_trials):
            X_noisy = X + np.random.randn(*X.shape) * noise_std
            predictions_noisy = self.model.predict(X_noisy)
            
            # Measure prediction difference
            diff = np.mean(np.abs(predictions_clean - predictions_noisy))
            prediction_errors.append(diff)
        
        mean_error = np.mean(prediction_errors)
        
        # Should have low prediction variance under noise
        passed = mean_error < 0.1
        
        return ValidationResult(
            test_name="robustness",
            passed=passed,
            score=1 - mean_error,
            message=f"Mean prediction difference under noise: {mean_error:.4f}",
            timestamp=datetime.now().isoformat(),
            metadata={'noise_std': noise_std, 'mean_error': float(mean_error)}
        )
    
    def comprehensive_validation(self, X_test: np.ndarray, y_test: np.ndarray,
                                X_train: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run comprehensive validation suite."""
        logger.info("Starting comprehensive model validation...")
        
        results = []
        
        # Data validation
        results.append(DataValidator.check_data_shape(X_test, y_test))
        results.append(DataValidator.check_data_quality(X_test))
        
        # Prediction validation
        y_pred = self.model.predict(X_test)
        results.append(self.validate_predictions(y_pred, y_test))
        
        # Reproducibility
        results.append(self.check_reproducibility(X_test[:100], n_runs=3))
        
        # OOD detection
        if X_train is not None:
            results.append(self.check_out_of_distribution(X_train, X_test))
        
        # Robustness
        results.append(self.check_robustness(X_test[:100]))
        
        self.results = results
        
        # Summary
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'passed': passed_count,
            'total': total_count,
            'pass_rate': passed_count / total_count,
            'results': [r.to_dict() for r in results]
        }
    
    def export_validation_report(self, output_path: Path) -> None:
        """Export validation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self.results),
            'passed_tests': sum(1 for r in self.results if r.passed),
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {output_path}")
    
    def report_validation_results(self) -> None:
        """Print validation results."""
        print("\n" + "="*60)
        print("MODEL VALIDATION REPORT")
        print("="*60)
        
        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"\n{status} | {result.test_name}")
            print(f"  Score: {result.score:.4f}")
            print(f"  Message: {result.message}")
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print(f"\n{'='*60}")
        print(f"TOTAL: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
        print("="*60 + "\n")


if __name__ == '__main__':
    logger.info("Model Validation Module Initialized")
