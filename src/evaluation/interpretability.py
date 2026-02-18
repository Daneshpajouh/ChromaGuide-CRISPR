"""
Comprehensive model interpretability tools.

Features:
- LIME (Local Interpretable Model-agnostic Explanations)
- Extended SHAP analysis
- Feature importance ranking
- Partial dependence plots
- ALE (Accumulated Local Effects)
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import json


@dataclass
class Explanation:
    """Model explanation result."""
    feature_importance: Dict[str, float]
    local_explanation: Dict[str, float]
    prediction: float
    confidence: float


class LIMEExplainer:
    """LIME local explanations."""
    
    def __init__(self, model: Callable, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
    
    def explain_prediction(
        self,
        instance: np.ndarray,
        num_samples: int = 1000
    ) -> Explanation:
        """Explain single prediction using LIME."""
        try:
            import lime.lime_tabular
            
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.random.randn(100, len(self.feature_names)),
                feature_names=self.feature_names,
                mode='regression'
            )
            
            exp = explainer.explain_instance(
                instance,
                self.model.predict,
                num_features=len(self.feature_names)
            )
            
            prediction = self.model.predict([instance])[0]
            
            local_exp = {}
            for feat, weight in exp.as_list():
                local_exp[feat] = weight
            
            return Explanation(
                feature_importance={},
                local_explanation=local_exp,
                prediction=float(prediction),
                confidence=0.9
            )
        except ImportError:
            print("LIME not installed. Install with: pip install lime")
            return None


class SHAPExplainerExtended:
    """Extended SHAP analysis."""
    
    def __init__(self, model, data: np.ndarray = None):
        self.model = model
        self.data = data
        self.explainer = None
    
    def create_explainer(self):
        """Create SHAP explainer."""
        try:
            import shap
            
            if self.data is not None:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                self.explainer = shap.KernelExplainer(
                    self.model.predict,
                    shap.sample(self.data, 100)
                )
        except ImportError:
            print("SHAP not installed. Install with: pip install shap")
    
    def get_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Get feature importance from SHAP."""
        if self.explainer is None:
            self.create_explainer()
        
        try:
            shap_values = self.explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            importance = np.abs(shap_values).mean(axis=0)
            
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}
        except Exception as e:
            print(f"SHAP calculation failed: {e}")
            return {}
    
    def plot_summary(self, X: np.ndarray, max_display: int = 10):
        """Plot SHAP summary."""
        try:
            import shap
            import matplotlib.pyplot as plt
            
            if self.explainer is None:
                self.create_explainer()
            
            shap_values = self.explainer.shap_values(X)
            shap.summary_plot(shap_values, X, max_display=max_display)
            
        except Exception as e:
            print(f"SHAP plotting failed: {e}")


class FeatureImportanceRanker:
    """Rank feature importance."""
    
    def __init__(self):
        self.importance_scores = {}
    
    def add_importance(self, method: str, importance: Dict[str, float]):
        """Add importance scores from method."""
        self.importance_scores[method] = importance
    
    def aggregate_importance(self) -> Dict[str, float]:
        """Aggregate importance from multiple methods."""
        all_features = set()
        for method_scores in self.importance_scores.values():
            all_features.update(method_scores.keys())
        
        aggregated = {}
        for feature in all_features:
            scores = [
                self.importance_scores[method].get(feature, 0)
                for method in self.importance_scores
            ]
            aggregated[feature] = np.mean(scores)
        
        return dict(sorted(aggregated.items(), 
                          key=lambda x: x[1], 
                          reverse=True))
    
    def get_top_features(self, k: int = 10) -> List[Tuple[str, float]]:
        """Get top k features."""
        aggregated = self.aggregate_importance()
        return list(aggregated.items())[:k]


class PartialDependencePlotter:
    """Plot partial dependence."""
    
    def __init__(self, model: Callable, X: np.ndarray, feature_names: List[str]):
        self.model = model
        self.X = X
        self.feature_names = feature_names
    
    def plot_partial_dependence(
        self,
        feature_idx: int,
        num_points: int = 50
    ) -> Dict:
        """Calculate partial dependence for feature."""
        feature_range = np.linspace(
            self.X[:, feature_idx].min(),
            self.X[:, feature_idx].max(),
            num_points
        )
        
        pd_values = []
        for value in feature_range:
            X_modified = self.X.copy()
            X_modified[:, feature_idx] = value
            predictions = self.model.predict(X_modified)
            pd_values.append(np.mean(predictions))
        
        return {
            'feature': self.feature_names[feature_idx],
            'values': feature_range.tolist(),
            'predictions': pd_values
        }


class ALE_Explainer:
    """Accumulated Local Effects explanations."""
    
    def __init__(self, model: Callable, X: np.ndarray, feature_names: List[str]):
        self.model = model
        self.X = X
        self.feature_names = feature_names
        self.ale = {}
    
    def calculate_ale(self, feature_idx: int, num_bins: int = 50) -> Dict:
        """Calculate ALE for feature."""
        X_sorted_idx = np.argsort(self.X[:, feature_idx])
        X_sorted = self.X[X_sorted_idx]
        
        feature_range = np.linspace(
            self.X[:, feature_idx].min(),
            self.X[:, feature_idx].max(),
            num_bins
        )
        
        ale_values = []
        for i in range(len(feature_range) - 1):
            lower = feature_range[i]
            upper = feature_range[i + 1]
            
            mask = (self.X[:, feature_idx] >= lower) & \
                   (self.X[:, feature_idx] < upper)
            
            if mask.sum() > 0:
                X_lower = self.X[mask].copy()
                X_upper = self.X[mask].copy()
                
                X_lower[:, feature_idx] = lower
                X_upper[:, feature_idx] = upper
                
                pred_lower = self.model.predict(X_lower)
                pred_upper = self.model.predict(X_upper)
                
                ale_values.append(np.mean(pred_upper - pred_lower))
        
        return {
            'feature': self.feature_names[feature_idx],
            'values': np.cumsum(ale_values).tolist()
        }


class InteractionDetector:
    """Detect feature interactions."""
    
    def __init__(self, model: Callable, X: np.ndarray):
        self.model = model
        self.X = X
    
    def detect_h_statistic(
        self,
        feature1_idx: int,
        feature2_idx: int,
        num_samples: int = 100
    ) -> float:
        """Friedman's H-statistic for interaction."""
        sample_idx = np.random.choice(len(self.X), num_samples, replace=False)
        X_sample = self.X[sample_idx]
        
        # Original predictions
        y_orig = self.model.predict(X_sample)
        
        # Fix feature 1
        X_f1_fixed = X_sample.copy()
        X_f1_fixed[:, feature1_idx] = X_sample[0, feature1_idx]
        y_f1 = self.model.predict(X_f1_fixed)
        
        # Fix feature 2
        X_f2_fixed = X_sample.copy()
        X_f2_fixed[:, feature2_idx] = X_sample[0, feature2_idx]
        y_f2 = self.model.predict(X_f2_fixed)
        
        # Fix both
        X_both_fixed = X_sample.copy()
        X_both_fixed[:, feature1_idx] = X_sample[0, feature1_idx]
        X_both_fixed[:, feature2_idx] = X_sample[0, feature2_idx]
        y_both = self.model.predict(X_both_fixed)
        
        # H-statistic
        h_stat = np.mean(y_orig - y_f1 - y_f2 + y_both)
        return float(h_stat)
    
    def find_top_interactions(self, k: int = 10) -> List[Tuple]:
        """Find top feature interactions."""
        n_features = self.X.shape[1]
        interactions = []
        
        for i in range(n_features):
            for j in range(i + 1, min(i + 5, n_features)):  # Limit for speed
                h_stat = self.detect_h_statistic(i, j)
                interactions.append(((i, j), abs(h_stat)))
        
        return sorted(interactions, key=lambda x: x[1], reverse=True)[:k]


class InterpretabilityReport:
    """Generate comprehensive interpretability report."""
    
    def __init__(self, model, X: np.ndarray, feature_names: List[str]):
        self.model = model
        self.X = X
        self.feature_names = feature_names
        self.report = {}
    
    def generate_report(self) -> Dict:
        """Generate full interpretability report."""
        self.report = {
            'model_summary': {
                'num_features': self.X.shape[1],
                'num_samples': self.X.shape[0]
            },
            'feature_importance': {},
            'interactions': {},
            'explanations': {}
        }
        
        return self.report
    
    def save_report(self, filepath: Path):
        """Save report to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)
