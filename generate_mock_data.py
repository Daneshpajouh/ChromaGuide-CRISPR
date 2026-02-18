#!/usr/bin/env python3
"""
Mock Data Generator for ChromaGuide Pipeline
=============================================

Generates realistic synthetic data for testing phases 2-4 without waiting for Phase 1.

Usage:
    python generate_mock_data.py --output data/mock/ --n_samples 1000
    
    Then phases 2-4 can be tested:
    python train_phase2_xgboost.py --data data/mock/crispro_features.pkl
"""

import argparse
import logging
import pickle
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockDataGenerator:
    """Generates realistic mock data for testing pipeline."""
    
    def __init__(self, output_dir: str = 'data/mock/', seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        np.random.seed(seed)
    
    def generate_crispro_features(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mock CRISPRO feature matrix and targets.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Feature matrix (n_samples, 100) and target values (n_samples,)
        """
        logger.info(f"Generating mock CRISPRO features: {n_samples} samples")
        
        # Features: combination of sequence features, biophysical, epigenomic
        n_features = 100
        
        X = np.random.randn(n_samples, n_features)
        
        # Make target correlated with features (some predictability)
        weights = np.random.randn(n_features)
        weights = weights / np.linalg.norm(weights)
        
        y = 0.5 + 0.3 * (X @ weights) + np.random.normal(0, 0.1, n_samples)
        y = np.clip(y, 0, 1)  # Clip to [0, 1]
        
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y shape: {y.shape}")
        logger.info(f"  y mean: {y.mean():.4f}, std: {y.std():.4f}")
        
        return X, y
    
    def generate_phase1_checkpoint(self, n_samples: int = 100) -> Dict:
        """Generate mock Phase 1 training checkpoint.
        
        Args:
            n_samples: Number of training samples
            
        Returns:
            Mock checkpoint dictionary
        """
        logger.info("Generating mock Phase 1 checkpoint")
        
        # Simulate training history
        epochs = 10
        training_loss = [0.25 - (i * 0.015) for i in range(epochs)]
        training_spearman = [0.45 + (i * 0.025) for i in range(epochs)]
        val_loss = [0.28 - (i * 0.014) for i in range(epochs)]
        val_spearman = [0.42 + (i * 0.022) for i in range(epochs)]
        
        checkpoint = {
            'epoch': epochs,
            'model_state_dict': None,  # Placeholder
            'optimizer_state_dict': None,
            'training_history': {
                'loss': training_loss,
                'spearman_r': training_spearman,
                'val_loss': val_loss,
                'val_spearman_r': val_spearman,
            },
            'best_val_spearman': max(val_spearman),
            'best_val_loss': min(val_loss),
            'timestamp': datetime.now().isoformat(),
        }
        
        logger.info(f"  Best validation Spearman: {checkpoint['best_val_spearman']:.4f}")
        logger.info(f"  Final training loss: {training_loss[-1]:.6f}")
        
        return checkpoint
    
    def generate_phase1_predictions(self, n_test: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mock Phase 1 test predictions.
        
        Args:
            n_test: Number of test samples
            
        Returns:
            Predictions and ground truth
        """
        logger.info(f"Generating mock Phase 1 predictions: {n_test} samples")
        
        # Real targets
        y_true = np.random.uniform(0, 1, n_test)
        
        # Predictions with good correlation
        y_pred = y_true + np.random.normal(0, 0.08, n_test)
        y_pred = np.clip(y_pred, 0, 1)
        
        return y_pred, y_true
    
    def generate_clinical_datasets(self) -> Dict:
        """Generate mock clinical validation datasets."""
        logger.info("Generating mock clinical validation datasets")
        
        datasets = {
            'circle_seq': {
                'n_sites': 500,
                'description': 'Real off-target sites from CIRCLE-seq',
                'y_true': np.random.binomial(1, 0.2, 500),  # 20% off-targets
            },
            'guide_seq': {
                'n_sites': 300,
                'description': 'Empirically mapped cleavage sites',
                'y_true': np.random.binomial(1, 0.15, 300),
            },
            'crispor': {
                'n_sites': 1000,
                'description': 'Predicted off-targets',
                'y_true': np.random.binomial(1, 0.1, 1000),
            }
        }
        
        for dataset_name, dataset_info in datasets.items():
            logger.info(f"  {dataset_name}: {dataset_info['n_sites']} sites, "
                       f"{dataset_info['y_true'].sum()} positives")
        
        return datasets
    
    def generate_evaluation_datasets(self) -> Dict:
        """Generate mock evaluation datasets for benchmarking."""
        logger.info("Generating mock evaluation datasets")
        
        evaluation_sets = {}
        
        datasets = [
            'crispro_main',
            'crispro_offtarget',
            'encode_gencode',
            'hct116_circulr',
            'k562_circulr',
            'gene_held_out',
            'cross_domain',
        ]
        
        for dataset_name in datasets:
            n_samples = np.random.randint(500, 2000)
            
            # Create synthetic targets
            y_true = np.random.uniform(0, 1, n_samples)
            
            # Create synthetic predictions for different models
            models = {}
            for model_id in range(1, 11):
                # Each model has different quality
                quality = 0.5 + (model_id * 0.05)  # Quality increases with model number
                noise = 1 - quality
                y_pred = y_true + np.random.normal(0, noise, n_samples)
                y_pred = np.clip(y_pred, 0, 1)
                models[f'model_{model_id}'] = y_pred
            
            evaluation_sets[dataset_name] = {
                'y_true': y_true,
                'predictions': models,
                'n_samples': n_samples,
            }
            
            logger.info(f"  {dataset_name}: {n_samples} samples")
        
        return evaluation_sets
    
    def save_all_mock_data(self, n_samples: int = 1000) -> None:
        """Generate and save all mock data."""
        logger.info("="*80)
        logger.info("MOCK DATA GENERATION")
        logger.info("="*80)
        
        # Phase 1 related
        X, y = self.generate_crispro_features(n_samples)
        
        # Save features for Phase 2
        features_data = {'X': X, 'y': y, 'feature_names': [f'feature_{i}' for i in range(100)]}
        with open(self.output_dir / 'crispro_features.pkl', 'wb') as f:
            pickle.dump(features_data, f)
        logger.info(f"✓ Saved: crispro_features.pkl")
        
        # Save full dataset for Phase 3
        with open(self.output_dir / 'crispro_dataset.pkl', 'wb') as f:
            pickle.dump(features_data, f)
        logger.info(f"✓ Saved: crispro_dataset.pkl")
        
        # Phase 1 checkpoint
        checkpoint = self.generate_phase1_checkpoint(n_samples)
        with open(self.output_dir / 'phase1_checkpoint.pkl', 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"✓ Saved: phase1_checkpoint.pkl")
        
        # Phase 1 predictions
        y_pred, y_true = self.generate_phase1_predictions(500)
        with open(self.output_dir / 'phase1_predictions.pkl', 'wb') as f:
            pickle.dump({'y_pred': y_pred, 'y_true': y_true}, f)
        logger.info(f"✓ Saved: phase1_predictions.pkl")
        
        # Clinical datasets
        clinical_data = self.generate_clinical_datasets()
        with open(self.output_dir / 'clinical_datasets.pkl', 'wb') as f:
            pickle.dump(clinical_data, f)
        logger.info(f"✓ Saved: clinical_datasets.pkl")
        
        # Evaluation datasets
        eval_data = self.generate_evaluation_datasets()
        with open(self.output_dir / 'evaluation_datasets.pkl', 'wb') as f:
            pickle.dump(eval_data, f)
        logger.info(f"✓ Saved: evaluation_datasets.pkl")
        
        # Summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'n_features': X.shape[1],
            'n_training_samples': n_samples,
            'n_test_samples': 500,
            'n_clinical_datasets': 3,
            'n_evaluation_datasets': 7,
            'files_generated': [
                'crispro_features.pkl',
                'crispro_dataset.pkl',
                'phase1_checkpoint.pkl',
                'phase1_predictions.pkl',
                'clinical_datasets.pkl',
                'evaluation_datasets.pkl',
            ]
        }
        
        with open(self.output_dir / 'mock_data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved: mock_data_summary.json")
        
        logger.info("\n" + "="*80)
        logger.info("Mock data generation complete!")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Files generated: {len(summary['files_generated'])}")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description='Generate mock data for testing')
    parser.add_argument('--output', type=str, default='data/mock/',
                       help='Output directory')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of training samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    generator = MockDataGenerator(output_dir=args.output, seed=args.seed)
    generator.save_all_mock_data(n_samples=args.n_samples)


if __name__ == '__main__':
    main()
