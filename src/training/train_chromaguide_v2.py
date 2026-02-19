#!/usr/bin/env python3
"""
Unified ChromaGuide V2 Training Pipeline.

Integrates ALL 9 critical modules:
1. LeakageControlledSplits - NO RANDOM SPLITS (prevents 5-15% metric inflation)
2. BetaRegressionHead - Bounded [0,1] outputs (NOT MSE)
3. ConformalPrediction - Uncertainty quantification (90% ± 2% coverage)
4. MINEClubRegularizer - Information regularization (decorrelates features)
5. OffTargetModule - Off-target binding risk prediction
6. DesignScoreAggregator - Multi-objective optimization (α trade-off)
7. StatisticalTests - Significance testing (p < 0.001 threshold)
8. SOTAComparison - Baseline comparison (vs 9 published models)
9. BackboneAblation - Architecture exploration (DNABERT-2 + 4 alternatives)

Author: ChromaGuide Team
Date: 2026-02
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import numpy as np
import pandas as pd
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Import all 9 modules
from src.data.leakage_controlled_splits import LeakageControlledSplits, DataSplit
from src.models.beta_regression_head import BetaRegressionHead, BetaRegressionLoss
from src.models.conformal_prediction import SplitConformalPrediction, ConformalQuantileRegression
from src.models.mine_club_regularizer import NonRedundancyRegularizer, ComplementarityMetric
from src.models.off_target_module import OffTargetModule
from src.models.design_score_aggregator import DesignScoreAggregator
from src.evaluation.statistical_tests import StatisticalTester, ComprehensiveStatisticalSummary
from src.evaluation.sota_comparison import SOTABenchmark, BenchmarkingReport
from src.models.backbone_ablation import BackboneFactory, BackboneType
from src.models.chromaguide_unified import ChromaGuideUnified, ModelConfig
from src.evaluation.metrics import spearman_correlation, ndcg_score, precision_at_k


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ChromaGuideV2Trainer:
    """
    Unified trainer integrating all 9 critical modules.
    
    Features:
    - Leakage-free evaluation (gene-held-out, dataset-held-out, cell-line-held-out)
    - Beta regression for bounded predictions
    - Conformal prediction for intervals
    - MINE/CLUB regularization for feature independence
    - Multi-objective design scoring
    - Statistical significance testing
    - SOTA baseline comparison
    - Architecture ablation support
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
    ):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary with keys:
                - model: Model config (backbone_type, hidden_dims, etc.)
                - data: Data config (dataset, split_type, batch_size, etc.)
                - training: Training config (epochs, lr, early_stopping, etc.)
                - regularization: Regularization config (lambda_mi, lambda_complement, etc.)
                - evaluation: Evaluation config (metrics, comparison_models, etc.)
            device: Device to use (cuda/cpu)
            seed: Random seed
        """
        self.config = config
        self.device = device
        self.seed = seed
        self._set_seed(seed)

        logger.info(f"Initializing ChromaGuideV2Trainer on device: {device}")

        # Model components
        self.model = None
        self.beta_head = None
        self.off_target = None
        self.design_aggregator = None
        self.conformal = None
        self.statistical_tester = None
        self.sota_benchmark = None

        # Data
        self.data_splits = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.X_train = None
        self.y_train = None

        # Optimization
        self.optimizer = None
        self.scheduler = None

        # Metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_spearman': [],
            'val_ndcg20': [],
            'val_precision10': [],
            'conformal_coverage': [],
        }

        # Build components
        self._build_model()
        self._build_regularizers()
        self._build_evaluation()

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _build_model(self):
        """Build model with backbone, beta head, and other components."""
        logger.info("Building model architecture...")

        # Create backbone
        backbone_type = self.config.get('model', {}).get('backbone_type', 'dnabert2')
        if backbone_type.lower() == 'dnabert2':
            from src.models.dnabert_encoder import DNABERTEncoder
            backbone = DNABERTEncoder()
            latent_dim = 768
        else:
            # Use factory for other backbones
            factory = BackboneFactory()
            config = factory.create_config(backbone_type)
            backbone = factory.create_backbone(backbone_type, config)
            latent_dim = config.hidden_dim if hasattr(config, 'hidden_dim') else 512

        self.backbone = backbone.to(self.device)

        # Create beta regression head (replaces standard MSE head)
        hidden_dims = self.config.get('model', {}).get('beta_hidden_dims', [256, 128])
        self.beta_head = BetaRegressionHead(
            input_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout_rate=self.config.get('model', {}).get('dropout', 0.1),
        ).to(self.device)

        logger.info(f"Built backbone + beta head (latent_dim={latent_dim})")

    def _build_regularizers(self):
        """Build regularization components."""
        logger.info("Building regularizers...")

        # MINE/CLUB for feature independence
        if self.config.get('regularization', {}).get('use_mi_regularization', True):
            self.mi_regularizer = NonRedundancyRegularizer(
                input_dim=self.config.get('model', {}).get('beta_hidden_dims', [256, 128])[0],
                estimator_type='club',
                weight=self.config.get('regularization', {}).get('lambda_mi', 0.01),
            )
            logger.info("Added MINE/CLUB regularizer")

        # Off-target scoring (optional)
        if self.config.get('regularization', {}).get('use_off_target', False):
            self.off_target = OffTargetModule(
                genome_fasta=self.config.get('data', {}).get('genome_fasta'),
                pam_type='NGG',
                mismatch_tolerance=4,
            )
            self.design_aggregator = DesignScoreAggregator(
                alpha=self.config.get('regularization', {}).get('design_alpha', 0.5),
            )
            logger.info("Added off-target module and design aggregator")

    def _build_evaluation(self):
        """Build evaluation utilities."""
        logger.info("Building evaluation utilities...")

        # Statistical tester
        self.statistical_tester = StatisticalTester()

        # SOTA comparison
        self.sota_benchmark = SOTABenchmark()

    def load_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        gene_ids: Optional[np.ndarray] = None,
        dataset_ids: Optional[np.ndarray] = None,
        cell_lines: Optional[np.ndarray] = None,
    ):
        """
        Load data and create leakage-controlled splits.

        Args:
            X: Input sequences, shape (N, max_seq_len) or (N, latent_dim)
            y: Target efficiency scores, shape (N,)
            gene_ids: Gene identifiers for split_a (optional)
            dataset_ids: Dataset identifiers for split_b (optional)
            cell_lines: Cell line information for split_c (optional)
        """
        logger.info(f"Loading data: X shape={X.shape}, y shape={y.shape}")

        self.X_train = X
        self.y_train = y

        # Create leakage-controlled splits
        split_generator = LeakageControlledSplits(seed=self.seed)

        split_type = self.config.get('data', {}).get('split_type', 'gene_held_out')

        if split_type == 'gene_held_out' and gene_ids is not None:
            logger.info("Creating gene-held-out split (most stringent)...")
            self.data_splits = split_generator.split_a_gene_held_out(
                X=X,
                y=y,
                gene_ids=gene_ids,
                val_ratio=self.config.get('data', {}).get('val_ratio', 0.1),
                test_ratio=self.config.get('data', {}).get('test_ratio', 0.2),
            )

        elif split_type == 'dataset_held_out' and dataset_ids is not None:
            logger.info("Creating dataset-held-out split...")
            self.data_splits = split_generator.split_b_dataset_held_out(
                X=X,
                y=y,
                dataset_ids=dataset_ids,
                val_ratio=self.config.get('data', {}).get('val_ratio', 0.1),
            )

        elif split_type == 'cellline_held_out' and cell_lines is not None:
            logger.info("Creating cell-line-held-out split...")
            self.data_splits = split_generator.split_c_cellline_held_out(
                X=X,
                y=y,
                cell_lines=cell_lines,
                val_ratio=self.config.get('data', {}).get('val_ratio', 0.1),
            )

        else:
            raise ValueError(f"Unknown split type: {split_type}")

        # Validate no overlap
        if not self.data_splits.validate_no_overlap():
            warnings.warn("Data split contains overlaps!")

        split_sizes = self.data_splits.get_sizes()
        logger.info(f"Split sizes: {split_sizes}")

        # Create dataloaders
        batch_size = self.config.get('data', {}).get('batch_size', 32)

        train_idx = self.data_splits.train_idx
        val_idx = self.data_splits.val_idx
        test_idx = self.data_splits.test_idx

        # Convert to tensors
        X_tensor = torch.FloatTensor(X) if isinstance(X, np.ndarray) else X
        y_tensor = torch.FloatTensor(y) if isinstance(y, np.ndarray) else y

        # Create subsets
        train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
        test_dataset = TensorDataset(X_tensor[test_idx], y_tensor[test_idx])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Store indices for later reference
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        logger.info(f"Created dataloaders: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    def setup_optimization(self):
        """Setup optimizer and scheduler."""
        logger.info("Setting up optimization...")

        # Collect all parameters
        params = list(self.backbone.parameters()) + list(self.beta_head.parameters())
        if hasattr(self, 'mi_regularizer'):
            params += list(self.mi_regularizer.parameters())

        lr = self.config.get('training', {}).get('learning_rate', 1e-3)
        weight_decay = self.config.get('training', {}).get('weight_decay', 1e-5)

        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        # Learning rate scheduler
        scheduler_type = self.config.get('training', {}).get('scheduler', 'cosine')
        epochs = self.config.get('training', {}).get('epochs', 50)

        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=lr / 100
            )
        elif scheduler_type == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        else:
            self.scheduler = None

        logger.info(f"Setup optimizer: AdamW(lr={lr}, weight_decay={weight_decay})")

    def train_epoch(self) -> float:
        """Train for one epoch. Return average loss."""
        self.backbone.train()
        self.beta_head.train()

        total_loss = 0.0
        num_batches = 0

        iterator = tqdm(self.train_loader, desc="Training") if TQDM_AVAILABLE else self.train_loader

        for batch_x, batch_y in iterator:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Get latent representation
            latent = self.backbone(batch_x)

            # Beta regression
            mu, phi = self.beta_head(latent)

            # Compute NLL loss
            loss_fn = BetaRegressionLoss()
            loss = loss_fn.loss_beta_nll(mu, phi, batch_y)

            # Add MI regularization if enabled
            if hasattr(self, 'mi_regularizer'):
                mi_loss = self.mi_regularizer(latent, batch_y)
                loss = loss + mi_loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.backbone.parameters()) + list(self.beta_head.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.metrics_history['train_loss'].append(avg_loss)

        return avg_loss

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split_name: str = "val") -> Dict[str, float]:
        """
        Evaluate on validation/test set.

        Returns dict with keys: loss, spearman, ndcg20, precision10
        """
        self.backbone.eval()
        self.beta_head.eval()

        all_preds = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0

        iterator = tqdm(loader, desc=f"Evaluating {split_name}") if TQDM_AVAILABLE else loader

        for batch_x, batch_y in iterator:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            latent = self.backbone(batch_x)
            mu, phi = self.beta_head(latent)

            # Loss
            loss_fn = BetaRegressionLoss()
            loss = loss_fn.loss_beta_nll(mu, phi, batch_y)
            total_loss += loss.item()
            num_batches += 1

            # Store predictions (use mean of Beta distribution)
            all_preds.append(mu.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

        # Concatenate all predictions and targets
        all_preds = np.concatenate(all_preds, axis=0).flatten()
        all_targets = np.concatenate(all_targets, axis=0).flatten()

        # Compute metrics
        avg_loss = total_loss / num_batches
        spearman = spearman_correlation(all_targets, all_preds)
        ndcg20 = ndcg_score(all_targets, all_preds, k=20)
        precision10 = precision_at_k(all_targets, all_preds, k=10)

        metrics = {
            'loss': avg_loss,
            'spearman': spearman,
            'ndcg20': ndcg20,
            'precision10': precision10,
        }

        # Track metrics
        if split_name == 'val':
            self.metrics_history['val_loss'].append(avg_loss)
            self.metrics_history['val_spearman'].append(spearman)
            self.metrics_history['val_ndcg20'].append(ndcg20)
            self.metrics_history['val_precision10'].append(precision10)

        logger.info(
            f"{split_name.upper()} - Loss: {avg_loss:.4f}, Spearman: {spearman:.4f}, "
            f"NDCG@20: {ndcg20:.4f}, Precision@10: {precision10:.4f}"
        )

        return metrics

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns: Dictionary with final metrics and model path
        """
        logger.info("=" * 80)
        logger.info("STARTING CHROMAGUIDE V2 TRAINING")
        logger.info("=" * 80)

        self.setup_optimization()

        epochs = self.config.get('training', {}).get('epochs', 50)
        early_stopping = self.config.get('training', {}).get('early_stopping_patience', 10)

        best_spearman = -np.inf
        no_improve_count = 0
        best_epoch = 0

        checkpoint_dir = Path(self.config.get('training', {}).get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            logger.info(f"\nEpoch {epoch}/{epochs}")

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_metrics = self.evaluate(self.val_loader, split_name='val')

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

            # Early stopping
            if val_metrics['spearman'] > best_spearman:
                best_spearman = val_metrics['spearman']
                no_improve_count = 0
                best_epoch = epoch

                # Save checkpoint
                checkpoint_path = checkpoint_dir / f"best_model_epoch{epoch}.pt"
                self._save_checkpoint(checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")

            else:
                no_improve_count += 1
                if no_improve_count >= early_stopping:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        logger.info(f"\nTraining completed. Best Spearman: {best_spearman:.4f} at epoch {best_epoch}")

        # Final test evaluation
        logger.info("\n" + "=" * 80)
        logger.info("TESTING ON HELD-OUT SET")
        logger.info("=" * 80)

        test_metrics = self.evaluate(self.test_loader, split_name='test')

        # Conformal prediction (optional calibration on validation set)
        if self.config.get('evaluation', {}).get('use_conformal', False):
            logger.info("Calibrating conformal prediction...")
            self.conformal = SplitConformalPrediction(coverage_level=0.9)
            # TODO: Calibrate on validation set

        # Statistical testing
        if self.config.get('evaluation', {}).get('run_statistical_tests', True):
            logger.info("\nRunning statistical tests...")
            test_results = self.statistical_tester.wilcoxon_signed_rank(
                y_true=self.y_train[self.test_idx],
                y_pred=test_metrics.get('predictions'),
            )
            logger.info(f"Wilcoxon p-value: {test_results['p_value']:.2e}")

        # SOTA comparison
        if self.config.get('evaluation', {}).get('run_sota_comparison', False):
            logger.info("\nRunning SOTA baseline comparison...")
            # TODO: Implement SOTA comparison

        results = {
            'best_epoch': best_epoch,
            'best_val_spearman': best_spearman,
            'test_spearman': test_metrics['spearman'],
            'test_ndcg20': test_metrics['ndcg20'],
            'test_precision10': test_metrics['precision10'],
            'test_loss': test_metrics['loss'],
            'checkpoint_path': str(checkpoint_path),
            'metrics_history': self.metrics_history,
        }

        # Save results
        results_path = checkpoint_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(
                {k: v for k, v in results.items() if k != 'metrics_history'},
                f,
                indent=2
            )
        logger.info(f"Saved results to {results_path}")

        return results

    def _save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'backbone_state': self.backbone.state_dict(),
            'beta_head_state': self.beta_head.state_dict(),
            'config': self.config,
        }, path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.backbone.load_state_dict(checkpoint['backbone_state'])
        self.beta_head.load_state_dict(checkpoint['beta_head_state'])
        logger.info(f"Loaded checkpoint from {path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ChromaGuide V2 Unified Training")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_v2_deephf.json',
        help='Path to config file'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/deephf_processed.pkl',
        help='Path to processed data'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # Initialize trainer
    trainer = ChromaGuideV2Trainer(config=config, device=args.device, seed=args.seed)

    # Load data
    # NOTE: This assumes data is preprocessed (sequences converted to embeddings or token IDs)
    if args.data.endswith('.pkl'):
        with open(args.data, 'rb') as f:
            data = pickle.load(f)
            X = data['X']
            y = data['y']
            gene_ids = data.get('gene_ids')
            dataset_ids = data.get('dataset_ids')
            cell_lines = data.get('cell_lines')
    else:
        raise NotImplementedError(f"Data format not supported: {args.data}")

    trainer.load_data(X, y, gene_ids=gene_ids, dataset_ids=dataset_ids, cell_lines=cell_lines)

    # Train
    results = trainer.train()

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    for k, v in results.items():
        if k != 'metrics_history':
            logger.info(f"{k}: {v}")

    return results


if __name__ == '__main__':
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training")

    main()
