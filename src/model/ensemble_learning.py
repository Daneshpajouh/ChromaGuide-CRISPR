"""
Advanced ensemble learning frameworks for CRISPR prediction.

Implements:
- Stacking/blending strategies
- Voting classifiers/regressors
- Bagging ensembles
- Boosting strategies
- Dynamic ensemble selection
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod
import warnings


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    voting_strategy: str = 'average'  # average, weighted, median
    use_stacking: bool = True
    meta_learner: str = 'ridge'
    cv_splits: int = 5
    optimal_ensemble_size: int = 5


class EnsembleMember:
    """Individual member in ensemble."""
    
    def __init__(self, model, name: str, weight: float = 1.0):
        self.model = model
        self.name = name
        self.weight = weight
        self.predictions = None
        self.train_predictions = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit member model."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X)
    
    def get_train_predictions(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Get training set predictions for stacking."""
        return self.model.predict(X)


class SimpleVotingEnsemble:
    """Simple voting ensemble."""
    
    def __init__(self, models: List, weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.weights = np.array(self.weights) / np.sum(self.weights)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all models."""
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted average predictions."""
        predictions = np.array([m.predict(X) for m in self.models])
        return np.average(predictions, axis=0, weights=self.weights)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted probability predictions."""
        if not hasattr(self.models[0], 'predict_proba'):
            raise ValueError("Models must have predict_proba method")
        proba = np.array([m.predict_proba(X) for m in self.models])
        return np.average(proba, axis=0, weights=self.weights)


class StackingEnsemble:
    """Stacking ensemble with meta-learner."""
    
    def __init__(self, base_models: List, meta_model, cv: int = 5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv = cv
        self.meta_features = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit stacking ensemble using cross-validation."""
        from sklearn.model_selection import KFold
        
        n_samples = len(X)
        n_base_models = len(self.base_models)
        
        # Create meta-features
        meta_X = np.zeros((n_samples, n_base_models))
        
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            for i, model in enumerate(self.base_models):
                model.fit(X_train, y_train)
                meta_X[val_idx, i] = model.predict(X_val)
        
        # Fit meta-learner
        self.meta_model.fit(meta_X, y)
        
        # Refit base models on full dataset
        for model in self.base_models:
            model.fit(X, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using stacking."""
        meta_X = np.column_stack([
            model.predict(X) for model in self.base_models
        ])
        return self.meta_model.predict(meta_X)


class BaggingEnsemble:
    """Bootstrap aggregating ensemble."""
    
    def __init__(self, base_model, n_estimators: int = 10, sample_fraction: float = 0.8):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.sample_fraction = sample_fraction
        self.models = []
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit bagging ensemble."""
        self.models = []
        n_samples = len(X)
        sample_size = int(n_samples * self.sample_fraction)
        
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, sample_size, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            from sklearn.base import clone
            model = clone(self.base_model)
            model.fit(X_sample, y_sample)
            self.models.append(model)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Average predictions from all base models."""
        predictions = np.array([m.predict(X) for m in self.models])
        return np.mean(predictions, axis=0)


class BoostingEnsemble:
    """Adaptive boosting ensemble."""
    
    def __init__(self, base_model, n_estimators: int = 10, learning_rate: float = 1.0):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.alphas = []
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit boosting ensemble."""
        self.models = []
        self.alphas = []
        
        sample_weights = np.ones(len(X)) / len(X)
        
        for _ in range(self.n_estimators):
            from sklearn.base import clone
            model = clone(self.base_model)
            
            # Sample according to weights
            indices = np.random.choice(
                len(X), len(X), replace=True, p=sample_weights
            )
            model.fit(X[indices], y[indices])
            
            # Calculate error
            predictions = model.predict(X)
            errors = (predictions != y).astype(int)
            error_rate = np.sum(sample_weights * errors)
            
            if error_rate >= 0.5 or error_rate == 0:
                break
            
            alpha = self.learning_rate * np.log((1 - error_rate) / error_rate)
            
            sample_weights *= np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)
            
            self.models.append(model)
            self.alphas.append(alpha)
        
        self.alphas = np.array(self.alphas)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble prediction."""
        predictions = np.array([m.predict(X) for m in self.models])
        weighted_sum = np.sum(self.alphas[:, np.newaxis] * predictions, axis=0)
        return np.sign(weighted_sum)


class DynamicEnsembleSelection:
    """Dynamically select ensemble members based on query sample."""
    
    def __init__(self, models: List, k_neighbors: int = 5):
        self.models = models
        self.k_neighbors = k_neighbors
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Store training data for DES."""
        self.X_train = X
        self.y_train = y
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using dynamically selected ensemble."""
        from sklearn.metrics.pairwise import euclidean_distances
        
        predictions = []
        for x in X:
            # Find k nearest neighbors
            distances = euclidean_distances([x], self.X_train)[0]
            knn_indices = np.argsort(distances)[:self.k_neighbors]
            
            # Select models with high accuracy on neighbors
            selected_models = []
            for model in self.models:
                knn_accuracy = np.mean(
                    model.predict(self.X_train[knn_indices]) == 
                    self.y_train[knn_indices]
                )
                if knn_accuracy > 0.5:  # Threshold
                    selected_models.append(model)
            
            if not selected_models:
                selected_models = self.models
            
            # Ensemble prediction
            preds = np.array([m.predict([x])[0] for m in selected_models])
            predictions.append(np.mean(preds))
        
        return np.array(predictions)


class RotationEnsemble:
    """Multi-view ensemble using rotated features."""
    
    def __init__(self, base_model, n_rotations: int = 5):
        self.base_model = base_model
        self.n_rotations = n_rotations
        self.models = []
        self.rotations = []
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit ensemble with feature rotations."""
        from sklearn.base import clone
        
        self.models = []
        self.rotations = []
        
        for _ in range(self.n_rotations):
            # Random rotation matrix
            from scipy.linalg import qr
            Q, _ = qr(np.random.randn(X.shape[1], X.shape[1]))
            self.rotations.append(Q)
            
            X_rotated = X @ Q
            model = clone(self.base_model)
            model.fit(X_rotated, y)
            self.models.append(model)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using rotation ensemble."""
        predictions = []
        for Q, model in zip(self.rotations, self.models):
            X_rotated = X @ Q
            predictions.append(model.predict(X_rotated))
        
        return np.mean(predictions, axis=0)


class EnsembleOptimizer:
    """Optimize ensemble weights and composition."""
    
    def __init__(self, models: List, config: EnsembleConfig):
        self.models = models
        self.config = config
        self.optimal_weights = None
    
    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """Find optimal weights using validation set."""
        from scipy.optimize import minimize
        
        predictions = np.array([m.predict(X_val) for m in self.models])
        
        def objective(weights):
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            mse = np.mean((ensemble_pred - y_val) ** 2)
            return mse
        
        initial_weights = np.ones(len(self.models)) / len(self.models)
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1)] * len(self.models)
        
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        self.optimal_weights = result.x
        return self.optimal_weights
    
    def select_best_subset(self, X_val: np.ndarray, y_val: np.ndarray) -> List:
        """Select best performing models."""
        from sklearn.metrics import r2_score
        
        scores = []
        for model in self.models:
            pred = model.predict(X_val)
            score = r2_score(y_val, pred)
            scores.append(score)
        
        top_indices = np.argsort(scores)[-self.config.optimal_ensemble_size:]
        return [self.models[i] for i in top_indices]
