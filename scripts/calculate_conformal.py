#!/usr/bin/env python3
"""Comprehensive Conformal Prediction Calibration for ChromaGuide PhD Proposal.

Implements split-conformal calibration with exchangeability testing
to meet PhD proposal deliverables:
- Split-conformal with calibration set
- Compute nonconformity scores on calibration set
- Find quantile for 90% coverage
- Report marginal coverage on test set (target: 0.88-0.92)
- Test exchangeability assumption

This script is designed to work with the trained ChromaGuide model
from jobs 56734059/56734060 and follows the methodology from
Chapter 5.3 of the ChromaGuide PhD Proposal.
"""

import torch
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr, ks_2samp
from sklearn.metrics import coverage_error
import matplotlib.pyplot as plt
import seaborn as sns
import json
from transformers import AutoTokenizer, AutoModel
import warnings

# Import ChromaGuide modules
import sys
import os
sys.path.insert(0, '/home/amird/chromaguide_experiments/src')
sys.path.insert(0, '/Users/studio/Desktop/PhD/Proposal/src')

try:
    from chromaguide.chromaguide_model import ChromaGuideModel
    from chromaguide.prediction_head import BetaRegressionHead, SplitConformalPredictor
    from chromaguide.conformal import BetaConformalPredictor
    from methods.conformal_prediction import SplitConformalPredictor as AdvancedConformalPredictor
    from methods.conformal_prediction import ConformalRegressionEvaluator
except ImportError as e:
    print(f"Warning: Could not import ChromaGuide modules: {e}")
    print("Running in standalone mode with built-in classes")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for PhD proposal targets
CONFORMAL_CONFIG = {
    'target_coverage': 0.90,  # 90% coverage target
    'coverage_tolerance': 0.02,  # ±2% tolerance (0.88-0.92)
    'exchangeability_alpha': 0.05,  # Significance level for exchangeability tests
    'quantiles_to_test': [0.85, 0.90, 0.95],  # Multiple coverage levels
    'bootstrap_resamples': 1000,  # For uncertainty estimation
}

class ExchangeabilityTester:
    """Tests the exchangeability assumption required for conformal prediction."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def test_distributional_exchangeability(self, calibration_scores, test_scores):
        """Test if calibration and test conformity scores come from same distribution."""
        statistic, p_value = ks_2samp(calibration_scores, test_scores)

        return {
            'test_name': 'Kolmogorov-Smirnov',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'reject_exchangeability': p_value < self.alpha,
            'interpretation': f"{'Reject' if p_value < self.alpha else 'Fail to reject'} exchangeability (p={p_value:.4f})"
        }

    def test_temporal_exchangeability(self, scores, timestamps=None):
        """Test for temporal dependence in conformity scores."""
        if timestamps is None:
            # Use indices as proxy for time
            timestamps = np.arange(len(scores))

        correlation, p_value = spearmanr(timestamps, scores)

        return {
            'test_name': 'Temporal Independence (Spearman)',
            'correlation': float(correlation),
            'p_value': float(p_value),
            'reject_exchangeability': p_value < self.alpha,
            'interpretation': f"{'Significant' if p_value < self.alpha else 'No significant'} temporal dependence (ρ={correlation:.4f}, p={p_value:.4f})"
        }

    def comprehensive_exchangeability_test(self, calibration_scores, test_scores):
        """Run comprehensive exchangeability tests."""
        results = {}

        # 1. Distributional test
        results['distributional'] = self.test_distributional_exchangeability(
            calibration_scores, test_scores
        )

        # 2. Temporal dependence in calibration scores
        results['temporal_calibration'] = self.test_temporal_exchangeability(calibration_scores)

        # 3. Temporal dependence in test scores
        results['temporal_test'] = self.test_temporal_exchangeability(test_scores)

        # Overall assessment
        any_violations = any([
            results['distributional']['reject_exchangeability'],
            results['temporal_calibration']['reject_exchangeability'],
            results['temporal_test']['reject_exchangeability']
        ])

        results['overall_assessment'] = {
            'exchangeability_violated': any_violations,
            'summary': "Exchangeability assumption violated" if any_violations else "Exchangeability assumption satisfied"
        }

        return results

class ComprehensiveConformalCalibrator:
    """Comprehensive conformal calibration meeting PhD proposal requirements."""

    def __init__(self, config=None):
        self.config = config or CONFORMAL_CONFIG
        self.target_coverage = self.config['target_coverage']
        self.alpha = 1 - self.target_coverage
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # State
        self.model = None
        self.tokenizer = None
        self.predictor = None
        self.exchangeability_tester = ExchangeabilityTester(
            alpha=self.config['exchangeability_alpha']
        )

        logger.info(f"Initialized conformal calibrator with target coverage: {self.target_coverage}")

    def load_model(self, model_path, model_type='chromaguide'):
        """Load trained ChromaGuide model."""
        try:
            if model_type == 'chromaguide':
                # Load full ChromaGuide model architecture
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
                logger.info(f"Loaded ChromaGuide model from {model_path}")

            elif model_type == 'dnabert_beta':
                # Load DNABERT-2 + Beta regression head setup
                MODEL_PATH = "zhihan1996/DNABERT-2-117M"
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
                backbone = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
                head = BetaRegressionHead(768)

                checkpoint = torch.load(model_path, map_location=self.device)
                backbone.load_state_dict(checkpoint['backbone_state_dict'])
                head.load_state_dict(checkpoint['head_state_dict'])

                self.model = {'backbone': backbone.to(self.device), 'head': head.to(self.device)}
                logger.info(f"Loaded DNABERT-2 + Beta head from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def prepare_data(self, data_path, calibration_fraction=0.2, test_fraction=0.2):
        """Prepare calibration and test sets with proper splits."""
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples from {data_path}")

        # Create proper splits for conformal calibration
        # Use stratified sampling to maintain label distribution

        # Sort by efficiency to stratify
        df_sorted = df.sort_values('efficiency')
        n = len(df_sorted)

        # Split indices maintaining distribution
        cal_size = int(n * calibration_fraction)
        test_size = int(n * test_fraction)
        train_size = n - cal_size - test_size

        # Interleaved sampling for better stratification
        indices = np.arange(n)
        cal_indices = indices[::int(1/calibration_fraction)][:cal_size]
        test_indices = indices[1::int(1/test_fraction)][:test_size]
        train_indices = np.setdiff1d(indices, np.concatenate([cal_indices, test_indices]))

        calibration_df = df_sorted.iloc[cal_indices].reset_index(drop=True)
        test_df = df_sorted.iloc[test_indices].reset_index(drop=True)
        train_df = df_sorted.iloc[train_indices].reset_index(drop=True)

        logger.info(f"Split sizes - Train: {len(train_df)}, Calibration: {len(calibration_df)}, Test: {len(test_df)}")

        return train_df, calibration_df, test_df

    def compute_predictions(self, sequences, batch_size=64):
        """Generate model predictions for sequences."""
        predictions = []
        uncertainties = []  # phi values for Beta distribution

        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i+batch_size]

                if isinstance(self.model, dict):
                    # DNABERT-2 + Beta head setup
                    tokens = self.tokenizer(
                        batch_seqs,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=30
                    ).to(self.device)

                    outputs = self.model['backbone'](
                        tokens['input_ids'],
                        tokens['attention_mask']
                    )
                    hidden = outputs.last_hidden_state.mean(dim=1)
                    beta_output = self.model['head'](hidden)

                    predictions.extend(beta_output['mu'].cpu().numpy().flatten())
                    uncertainties.extend(beta_output['phi'].cpu().numpy().flatten())

                else:
                    # Full ChromaGuide model
                    # This would need to be implemented based on the actual model interface
                    raise NotImplementedError("Full ChromaGuide model prediction not implemented")

        return np.array(predictions), np.array(uncertainties)

    def calibrate_conformal_predictor(self, calibration_df, test_df):
        """Perform split-conformal calibration."""
        logger.info("Starting conformal calibration...")

        # Get predictions for calibration set
        cal_sequences = calibration_df['sequence'].tolist()
        cal_labels = calibration_df['efficiency'].values

        cal_predictions, cal_uncertainties = self.compute_predictions(cal_sequences)

        # Initialize and calibrate predictor
        try:
            # Try using the advanced conformal predictor if available
            self.predictor = AdvancedConformalPredictor(self.model, alpha=self.alpha)

            # Convert to tensors for calibration
            cal_features = torch.tensor(cal_predictions).float().unsqueeze(-1)
            cal_targets = torch.tensor(cal_labels).float()

            self.predictor.calibrate(cal_features, cal_targets)

        except:
            # Fallback to built-in predictor
            logger.info("Using built-in Beta conformal predictor")
            self.predictor = BetaConformalPredictor(alpha=self.alpha)
            self.predictor.calibrate(cal_predictions, cal_uncertainties, cal_labels)

        # Compute conformity scores for exchangeability testing
        if hasattr(self.predictor, 'calibration_scores'):
            cal_conformity_scores = self.predictor.calibration_scores
        else:
            # Compute manually for Beta predictor
            cal_conformity_scores = self.predictor.compute_conformity_scores(
                cal_predictions, cal_uncertainties, cal_labels
            )

        logger.info(f"Calibration complete with {len(cal_conformity_scores)} conformity scores")

        # Get test predictions and compute test conformity scores
        test_sequences = test_df['sequence'].tolist()
        test_labels = test_df['efficiency'].values
        test_predictions, test_uncertainties = self.compute_predictions(test_sequences)

        # Compute test conformity scores for exchangeability testing
        test_conformity_scores = self.predictor.compute_conformity_scores(
            test_predictions, test_uncertainties, test_labels
        )

        return {
            'calibration_scores': cal_conformity_scores,
            'test_scores': test_conformity_scores,
            'calibration_size': len(cal_conformity_scores),
            'quantile_value': np.quantile(cal_conformity_scores, 1 - self.alpha),
            'alpha': self.alpha,
            'target_coverage': self.target_coverage
        }

    def evaluate_coverage(self, test_df):
        """Evaluate conformal prediction coverage on test set."""
        test_sequences = test_df['sequence'].tolist()
        test_labels = test_df['efficiency'].values

        test_predictions, test_uncertainties = self.compute_predictions(test_sequences)

        # Get prediction intervals
        if hasattr(self.predictor, 'predict'):
            # Advanced predictor
            test_features = torch.tensor(test_predictions).float().unsqueeze(-1)
            point_preds, lower_bounds, upper_bounds = self.predictor.predict(test_features)

            point_preds = point_preds.numpy().flatten()
            lower_bounds = lower_bounds.numpy().flatten()
            upper_bounds = upper_bounds.numpy().flatten()

        else:
            # Beta predictor
            point_preds = test_predictions
            lower_bounds, upper_bounds = self.predictor.predict_intervals(
                test_predictions, test_uncertainties
            )

        # Calculate coverage metrics
        coverage = np.mean((test_labels >= lower_bounds) & (test_labels <= upper_bounds))
        interval_widths = upper_bounds - lower_bounds
        mean_width = np.mean(interval_widths)

        # Calculate coverage by efficiency quantiles
        efficiency_quantiles = np.quantile(test_labels, [0.25, 0.5, 0.75])
        quantile_coverage = {}

        for i, q in enumerate([0.25, 0.5, 0.75]):
            if i == 0:
                mask = test_labels <= efficiency_quantiles[0]
                name = f'Q1 (≤{efficiency_quantiles[0]:.3f})'
            elif i == 1:
                mask = (test_labels > efficiency_quantiles[0]) & (test_labels <= efficiency_quantiles[1])
                name = f'Q2 ({efficiency_quantiles[0]:.3f}-{efficiency_quantiles[1]:.3f})'
            else:
                mask = test_labels > efficiency_quantiles[1]
                name = f'Q3-Q4 (>{efficiency_quantiles[1]:.3f})'

            if mask.sum() > 0:
                quantile_coverage[name] = np.mean(
                    (test_labels[mask] >= lower_bounds[mask]) &
                    (test_labels[mask] <= upper_bounds[mask])
                )

        return {
            'overall_coverage': float(coverage),
            'mean_interval_width': float(mean_width),
            'target_coverage': self.target_coverage,
            'coverage_within_tolerance': abs(coverage - self.target_coverage) <= self.config['coverage_tolerance'],
            'coverage_by_efficiency_quantile': quantile_coverage,
            'interval_statistics': {
                'min_width': float(np.min(interval_widths)),
                'max_width': float(np.max(interval_widths)),
                'std_width': float(np.std(interval_widths))
            },
            'prediction_statistics': {
                'mae': float(np.mean(np.abs(point_preds - test_labels))),
                'rmse': float(np.sqrt(np.mean((point_preds - test_labels)**2))),
                'spearman_rho': float(spearmanr(point_preds, test_labels)[0])
            }
        }

    def test_multiple_coverage_levels(self, test_df):
        """Test conformal prediction at multiple coverage levels."""
        results = {}

        for target_coverage in self.config['quantiles_to_test']:
            alpha = 1 - target_coverage

            # Create temporary predictor for this coverage level
            if hasattr(self.predictor, 'alpha'):
                original_alpha = self.predictor.alpha
                self.predictor.alpha = alpha

                coverage_results = self.evaluate_coverage(test_df)
                coverage_results['target_coverage'] = target_coverage
                results[f'coverage_{target_coverage}'] = coverage_results

                # Restore original alpha
                self.predictor.alpha = original_alpha

        return results

    def generate_calibration_report(self, calibration_results, coverage_results,
                                  exchangeability_results, output_dir):
        """Generate comprehensive calibration report."""

        report = {
            'experiment_info': {
                'target_coverage': self.target_coverage,
                'tolerance': self.config['coverage_tolerance'],
                'calibration_size': calibration_results['calibration_size'],
                'model_device': str(self.device),
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'calibration_summary': {
                'alpha': calibration_results['alpha'],
                'quantile_value': float(calibration_results['quantile_value']),
                'conformal_calibrated': True
            },
            'coverage_evaluation': coverage_results,
            'exchangeability_tests': exchangeability_results,
            'phd_proposal_targets': {
                'target_coverage_range': '0.88-0.92',
                'achieved_coverage': coverage_results['overall_coverage'],
                'target_met': coverage_results['coverage_within_tolerance'],
                'exchangeability_satisfied': not exchangeability_results['overall_assessment']['exchangeability_violated']
            }
        }

        # Save detailed report
        output_path = output_dir / 'conformal_calibration_report.json'
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate summary for PhD proposal
        self.generate_proposal_summary(report, output_dir)

        logger.info(f"Conformal calibration report saved to {output_path}")

        return report

    def generate_proposal_summary(self, report, output_dir):
        """Generate PhD proposal-ready summary."""

        coverage = report['coverage_evaluation']['overall_coverage']
        target_met = report['phd_proposal_targets']['target_met']
        exchangeable = report['phd_proposal_targets']['exchangeability_satisfied']

        summary = f"""
# ChromaGuide Conformal Prediction Evaluation - PhD Proposal Results

## Key Results

**Target:** Conformal coverage within ±0.02 of 0.90 under exchangeability assumption

### Coverage Performance
- **Achieved Coverage:** {coverage:.3f} ({coverage*100:.1f}%)
- **Target Range:** 0.88-0.92 (88%-92%)
- **Target Met:** {'✓ YES' if target_met else '✗ NO'}
- **Deviation:** {abs(coverage - 0.90)*100:+.1f}% from target

### Exchangeability Assessment
- **Assumption Satisfied:** {'✓ YES' if exchangeable else '✗ NO'}
- **Distributional Test:** {report['exchangeability_tests']['distributional']['interpretation']}
- **Temporal Independence:** {report['exchangeability_tests']['temporal_calibration']['interpretation']}

### Statistical Summary
- **Spearman ρ:** {report['coverage_evaluation']['prediction_statistics']['spearman_rho']:.3f}
- **Mean Interval Width:** {report['coverage_evaluation']['mean_interval_width']:.3f}
- **Calibration Set Size:** {report['experiment_info']['calibration_size']}

### PhD Proposal Compliance
{'✓ All targets met - ready for proposal defense' if (target_met and exchangeable) else '⚠ Some targets not met - requires investigation'}

Generated: {report['experiment_info']['timestamp']}
        """

        with open(output_dir / 'conformal_summary_phd_proposal.md', 'w') as f:
            f.write(summary.strip())

    def plot_calibration_diagnostics(self, calibration_results, coverage_results,
                                   test_df, output_dir):
        """Generate calibration diagnostic plots."""

        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Conformity score distribution
        ax1 = axes[0, 0]
        cal_scores = calibration_results['calibration_scores']
        test_scores = calibration_results['test_scores']

        ax1.hist(cal_scores, bins=30, alpha=0.7, label='Calibration', density=True)
        ax1.hist(test_scores, bins=30, alpha=0.7, label='Test', density=True)
        ax1.axvline(calibration_results['quantile_value'], color='red', linestyle='--',
                   label=f'{self.target_coverage*100}% Quantile')
        ax1.set_xlabel('Conformity Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Conformity Score Distribution')
        ax1.legend()

        # Plot 2: Coverage by efficiency quantile
        ax2 = axes[0, 1]
        quantile_cov = coverage_results['coverage_by_efficiency_quantile']
        quantiles = list(quantile_cov.keys())
        coverages = list(quantile_cov.values())

        bars = ax2.bar(range(len(quantiles)), coverages, alpha=0.7)
        ax2.axhline(self.target_coverage, color='red', linestyle='--', label='Target')
        ax2.axhline(self.target_coverage - self.config['coverage_tolerance'],
                   color='orange', linestyle=':', label='Tolerance')
        ax2.axhline(self.target_coverage + self.config['coverage_tolerance'],
                   color='orange', linestyle=':', alpha=0.7)
        ax2.set_xticks(range(len(quantiles)))
        ax2.set_xticklabels(quantiles, rotation=45)
        ax2.set_ylabel('Coverage')
        ax2.set_title('Coverage by Efficiency Quantile')
        ax2.legend()

        # Plot 3: Prediction vs True values
        ax3 = axes[1, 0]
        test_sequences = test_df['sequence'].tolist()
        test_labels = test_df['efficiency'].values
        test_predictions, _ = self.compute_predictions(test_sequences)

        ax3.scatter(test_labels, test_predictions, alpha=0.6, s=20)
        min_val = min(test_labels.min(), test_predictions.min())
        max_val = max(test_labels.max(), test_predictions.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax3.set_xlabel('True Efficiency')
        ax3.set_ylabel('Predicted Efficiency')
        ax3.set_title(f'Prediction Quality (ρ={coverage_results["prediction_statistics"]["spearman_rho"]:.3f})')

        # Plot 4: Interval width distribution
        ax4 = axes[1, 1]
        widths = coverage_results['interval_statistics']
        ax4.hist([widths['min_width'], widths['max_width'], widths['std_width']],
                bins=20, alpha=0.7)
        ax4.axvline(coverage_results['mean_interval_width'], color='red', linestyle='--',
                   label=f'Mean: {coverage_results["mean_interval_width"]:.3f}')
        ax4.set_xlabel('Interval Width')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Prediction Interval Width Distribution')
        ax4.legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'conformal_calibration_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Diagnostic plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='ChromaGuide Conformal Prediction Calibration')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained ChromaGuide model')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to evaluation dataset CSV')
    parser.add_argument('--output-dir', type=str, default='results/conformal_calibration',
                       help='Output directory for results')
    parser.add_argument('--model-type', type=str, default='dnabert_beta',
                       choices=['chromaguide', 'dnabert_beta'],
                       help='Type of model to load')
    parser.add_argument('--target-coverage', type=float, default=0.90,
                       help='Target coverage level (default: 0.90)')
    parser.add_argument('--calibration-fraction', type=float, default=0.2,
                       help='Fraction of data for calibration (default: 0.2)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update config with command line arguments
    config = CONFORMAL_CONFIG.copy()
    config['target_coverage'] = args.target_coverage

    # Initialize calibrator
    calibrator = ComprehensiveConformalCalibrator(config)

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    calibrator.load_model(args.model_path, args.model_type)

    # Prepare data splits
    logger.info(f"Preparing data from {args.data_path}")
    train_df, calibration_df, test_df = calibrator.prepare_data(
        args.data_path,
        calibration_fraction=args.calibration_fraction
    )

    # Perform conformal calibration
    logger.info("Performing conformal calibration...")
    calibration_results = calibrator.calibrate_conformal_predictor(calibration_df, test_df)

    # Evaluate coverage
    logger.info("Evaluating coverage on test set...")
    coverage_results = calibrator.evaluate_coverage(test_df)

    # Test exchangeability assumption
    logger.info("Testing exchangeability assumption...")
    exchangeability_results = calibrator.exchangeability_tester.comprehensive_exchangeability_test(
        calibration_results['calibration_scores'],
        calibration_results['test_scores']
    )

    # Test multiple coverage levels
    logger.info("Testing multiple coverage levels...")
    multi_coverage_results = calibrator.test_multiple_coverage_levels(test_df)

    # Generate comprehensive report
    logger.info("Generating calibration report...")
    report = calibrator.generate_calibration_report(
        calibration_results, coverage_results, exchangeability_results, output_dir
    )

    # Generate diagnostic plots
    logger.info("Generating diagnostic plots...")
    calibrator.plot_calibration_diagnostics(
        calibration_results, coverage_results, test_df, output_dir
    )

    # Print summary
    print("\n" + "="*60)
    print("CHROMAGUIDE CONFORMAL CALIBRATION COMPLETE")
    print("="*60)
    print(f"Target Coverage: {config['target_coverage']*100:.1f}%")
    print(f"Achieved Coverage: {coverage_results['overall_coverage']*100:.1f}%")
    print(f"Within Tolerance: {'✓' if coverage_results['coverage_within_tolerance'] else '✗'}")
    print(f"Exchangeability OK: {'✓' if not exchangeability_results['overall_assessment']['exchangeability_violated'] else '✗'}")
    print(f"Results saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
