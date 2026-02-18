#!/usr/bin/env python3
"""
Phase 2: CRISPRO-XGBoost Benchmarking
======================================

Trains XGBoost ensemble on CRISPRO dataset to establish competitive baseline.
Includes feature engineering, hyperparameter optimization, and cross-validation.

Usage:
    python train_phase2_xgboost.py \
        --data data/processed/crispro_features.pkl \
        --output checkpoints/phase2_xgboost/ \
        --n_trials 100
"""

import argparse
import logging
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Any

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CRISPROXGBoostBenchmark:
    """XGBoost baseline for CRISPRO dataset."""
    
    def __init__(self, output_dir: str = 'checkpoints/phase2_xgboost/', seed: int = 42):
        """Initialize XGBoost benchmark.
        
        Args:
            output_dir: Directory to save results
            seed: Random seed
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        np.random.seed(seed)
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.best_params = None
        self.cv_results = {}
        self.test_results = {}
        
        self.log_file = self.output_dir / f'phase2_xgboost_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load preprocessed CRISPRO features and targets.
        
        Args:
            data_path: Path to pickle file with features
            
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            feature_names: List of feature names
        """
        logger.info(f"Loading data from {data_path}")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        X = data.get('X', data.get('features'))
        y = data.get('y', data.get('targets'))
        feature_names = data.get('feature_names', [f'feature_{i}' for i in range(X.shape[1])])
        
        logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
        logger.info(f"Target statistics: mean={y.mean():.4f}, std={y.std():.4f}, " 
                   f"min={y.min():.4f}, max={y.max():.4f}")
        
        self.feature_names = feature_names
        return X, y, feature_names
    
    def objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Cross-validation RMSE
        """
        # Hyperparameter space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': self.seed,
            'n_jobs': -1,
            'verbosity': 0,
        }
        
        # Cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        rmse_scores = []
        
        for train_idx, val_idx in kfold.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = xgb.XGBRegressor(**params, tree_method='hist')
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmse_scores.append(rmse)
        
        mean_rmse = np.mean(rmse_scores)
        logger.info(f"Trial {trial.number}: params={params}, RMSE={mean_rmse:.4f}")
        
        return mean_rmse
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                                 n_trials: int = 100) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training targets
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters dictionary
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.seed),
            pruner=MedianPruner()
        )
        
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        logger.info(f"Best hyperparameters: {self.best_params}")
        
        # Save optimization history
        trials_df = study.trials_dataframe()
        trials_df.to_csv(self.output_dir / 'optimization_history.csv', index=False)
        
        return self.best_params
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              optimize: bool = True, n_trials: int = 100) -> None:
        """Train XGBoost model on data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            optimize: Whether to run hyperparameter optimization
            n_trials: Number of optimization trials
        """
        logger.info("Starting Phase 2 XGBoost training")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Optimize hyperparameters if requested
        if optimize:
            best_params = self.optimize_hyperparameters(X_train_scaled, y_train, n_trials)
        else:
            best_params = {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 1,
                'min_child_weight': 3,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            }
            logger.info(f"Using default hyperparameters: {best_params}")
        
        # Train final model
        self.model = xgb.XGBRegressor(
            **best_params,
            n_jobs=-1,
            random_state=self.seed,
        )
        
        logger.info("Training final model on full training set")
        self.model.fit(X_train_scaled, y_train)
        
        # Save model
        model_path = self.output_dir / 'xgboost_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Saved model to {model_path}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv(
            self.output_dir / 'feature_importance.csv',
            index=False
        )
        
        logger.info(f"\nTop 10 Features:\n{feature_importance.head(10)}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating on test set")
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        spearman_r, spearman_p = spearmanr(y_test, y_pred)
        pearson_r, pearson_p = pearsonr(y_test, y_pred)
        
        self.test_results = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'n_test_samples': len(y_test),
        }
        
        logger.info(f"\nTest Results:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  Spearman r: {spearman_r:.4f} (p={spearman_p:.2e})")
        logger.info(f"  Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})")
        
        return self.test_results
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, List[float]]:
        """Perform k-fold cross-validation.
        
        Args:
            X: Features
            y: Targets
            n_splits: Number of folds
            
        Returns:
            Dictionary of CV metrics
        """
        logger.info(f"Running {n_splits}-fold cross-validation")
        
        X_scaled = self.scaler.fit_transform(X)
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        
        results = {
            'mse': [],
            'rmse': [],
            'mae': [],
            'spearman_r': [],
            'pearson_r': [],
        }
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            
            results['mse'].append(mean_squared_error(y_val, y_pred))
            results['rmse'].append(np.sqrt(results['mse'][-1]))
            results['mae'].append(mean_absolute_error(y_val, y_pred))
            results['spearman_r'].append(spearmanr(y_val, y_pred)[0])
            results['pearson_r'].append(pearsonr(y_val, y_pred)[0])
            
            logger.info(f"Fold {fold+1}: RMSE={results['rmse'][-1]:.4f}, "
                       f"Spearman r={results['spearman_r'][-1]:.4f}")
        
        # Compute aggregates
        for key in results:
            results[key] = {
                'values': [float(v) for v in results[key]],
                'mean': float(np.mean(results[key])),
                'std': float(np.std(results[key])),
            }
        
        self.cv_results = results
        logger.info(f"\nCV Results: RMSE={results['rmse']['mean']:.4f}±{results['rmse']['std']:.4f}")
        
        return results
    
    def save_results(self) -> None:
        """Save all results to JSON."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_results': self.test_results,
            'cv_results': self.cv_results,
            'best_hyperparameters': self.best_params,
        }
        
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Phase 2: CRISPRO XGBoost Benchmark')
    parser.add_argument('--data', type=str, required=True, help='Path to processed data pickle')
    parser.add_argument('--output', type=str, default='checkpoints/phase2_xgboost/', 
                       help='Output directory')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_split', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--no_optimize', action='store_true', help='Skip hyperparameter optimization')
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = CRISPROXGBoostBenchmark(output_dir=args.output, seed=args.seed)
    
    # Load data
    X, y, feature_names = benchmark.load_data(args.data)
    
    # Split train/test
    n_test = int(len(X) * args.test_split)
    test_idx = np.random.choice(len(X), n_test, replace=False)
    train_idx = np.array([i for i in range(len(X)) if i not in test_idx])
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train model
    benchmark.train(X_train, y_train, optimize=not args.no_optimize, n_trials=args.n_trials)
    
    # Evaluate
    benchmark.evaluate(X_test, y_test)
    
    # Cross-validation
    benchmark.cross_validate(X_train, y_train, n_splits=5)
    
    # Save results
    benchmark.save_results()
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 2 COMPLETE: XGBoost benchmark ready for comparison")
    logger.info("="*80)


if __name__ == '__main__':
    main()
