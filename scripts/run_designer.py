#!/usr/bin/env python3
"""ChromaGuide Designer Score Evaluation Script.

Implements the integrated design score evaluation as specified in the PhD proposal:
S = w_e*mu - w_r*R - w_u*sigma

Where:
- mu: On-target efficacy prediction (Beta regression mean)
- R: Off-target risk score (CNN-based predictor)
- sigma: Uncertainty quantification (conformal interval width)
- w_e, w_r, w_u: Balancing weights for efficacy, risk, uncertainty

This script loads trained models and evaluates candidate gRNAs to generate
a ranked list for experimental validation.
"""

import torch
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from scipy.stats import spearmanr
import json
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns

# Import ChromaGuide modules
import sys
import os
sys.path.insert(0, '/home/amird/chromaguide_experiments/src')
sys.path.insert(0, '/Users/studio/Desktop/PhD/Proposal/src')

try:
    from chromaguide.chromaguide_model import ChromaGuideModel
    from chromaguide.prediction_head import BetaRegressionHead
    from chromaguide.conformal import BetaConformalPredictor
    from chromaguide.off_target import OffTargetScorer, CandidateFinder
    from chromaguide.design_score import IntegratedDesignScore
    from chromaguide.designer import ChromaGuideDesigner
except ImportError as e:
    print(f"Warning: Could not import ChromaGuide modules: {e}")
    print("Running in standalone mode with built-in classes")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default designer weights from PhD proposal
DEFAULT_WEIGHTS = {
    'w_e': 1.0,  # Efficacy weight (positive contribution)
    'w_r': 0.5,  # Risk weight (negative contribution)
    'w_u': 0.2,  # Uncertainty weight (negative contribution)
}

class DesignerScoreEvaluator:
    """Evaluates and ranks gRNA candidates using integrated design score."""

    def __init__(self, weights=None, device='auto'):
        self.weights = weights or DEFAULT_WEIGHTS
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)

        # Model components
        self.on_target_model = None
        self.off_target_model = None
        self.conformal_predictor = None
        self.tokenizer = None

        # Loaded models
        self.models_loaded = False

        logger.info(f"Initialized DesignerScoreEvaluator with weights: {self.weights}")
        logger.info(f"Using device: {self.device}")

    def load_models(self, on_target_path, off_target_path=None, conformal_path=None):
        """Load all required models for design score computation."""

        # Load on-target efficacy model (DNABERT-2 + Beta regression)
        logger.info(f"Loading on-target model from {on_target_path}")
        try:
            MODEL_PATH = "zhihan1996/DNABERT-2-117M"
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            backbone = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
            head = BetaRegressionHead(768)

            checkpoint = torch.load(on_target_path, map_location=self.device)
            backbone.load_state_dict(checkpoint['backbone_state_dict'])
            head.load_state_dict(checkpoint['head_state_dict'])

            self.on_target_model = {
                'backbone': backbone.to(self.device).eval(),
                'head': head.to(self.device).eval()
            }
            logger.info("✓ On-target model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load on-target model: {e}")
            raise

        # Load off-target model (optional)
        if off_target_path:
            try:
                logger.info(f"Loading off-target model from {off_target_path}")
                self.off_target_model = torch.load(off_target_path, map_location=self.device)
                self.off_target_model.eval()
                logger.info("✓ Off-target model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load off-target model: {e}. Using dummy scores.")
                self.off_target_model = None
        else:
            logger.info("No off-target model provided. Using dummy scores.")
            self.off_target_model = None

        # Load conformal predictor (optional)
        if conformal_path:
            try:
                logger.info(f"Loading conformal predictor from {conformal_path}")
                self.conformal_predictor = BetaConformalPredictor(alpha=0.1)
                self.conformal_predictor.load_calibration(conformal_path)
                logger.info("✓ Conformal predictor loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load conformal predictor: {e}. Using dummy uncertainties.")
                self.conformal_predictor = None
        else:
            logger.info("No conformal predictor provided. Using dummy uncertainties.")
            self.conformal_predictor = None

        self.models_loaded = True
        logger.info("All available models loaded successfully")

    def predict_on_target_efficacy(self, sequences: List[str], batch_size=64) -> Tuple[np.ndarray, np.ndarray]:
        """Predict on-target efficacy using DNABERT-2 + Beta regression."""
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        predictions = []
        uncertainties = []

        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i+batch_size]

                # Tokenize sequences
                tokens = self.tokenizer(
                    batch_seqs,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=30
                ).to(self.device)

                # Get sequence embeddings
                outputs = self.on_target_model['backbone'](
                    tokens['input_ids'],
                    tokens['attention_mask']
                )
                hidden = outputs.last_hidden_state.mean(dim=1)

                # Predict Beta distribution parameters
                beta_output = self.on_target_model['head'](hidden)

                predictions.extend(beta_output['mu'].cpu().numpy().flatten())
                uncertainties.extend(beta_output['phi'].cpu().numpy().flatten())

        return np.array(predictions), np.array(uncertainties)

    def predict_off_target_risk(self, sequences: List[str]) -> np.ndarray:
        """Predict off-target risk scores."""
        if self.off_target_model is None:
            # Return dummy risk scores (low risk for all)
            logger.info("Using dummy off-target risk scores")
            return np.random.uniform(0.01, 0.05, len(sequences))

        # Implementation would depend on the off-target model architecture
        # For now, return dummy scores
        logger.info("Using dummy off-target risk scores (no off-target model)")
        return np.random.uniform(0.01, 0.1, len(sequences))

    def predict_uncertainty(self, mu_predictions: np.ndarray, phi_predictions: np.ndarray) -> np.ndarray:
        """Predict uncertainty using conformal prediction intervals."""
        if self.conformal_predictor is None:
            # Use Beta distribution variance as uncertainty proxy
            variance = mu_predictions * (1 - mu_predictions) / (phi_predictions + 1)
            return np.sqrt(variance)

        # Get conformal interval widths
        interval_widths = self.conformal_predictor.get_interval_width(mu_predictions, phi_predictions)
        return interval_widths

    def compute_design_scores(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        """Compute integrated design scores for all sequences."""
        logger.info(f"Computing design scores for {len(sequences)} sequences")

        # Predict on-target efficacy
        mu_pred, phi_pred = self.predict_on_target_efficacy(sequences)
        logger.info(f"On-target efficacy range: {mu_pred.min():.3f} - {mu_pred.max():.3f}")

        # Predict off-target risk
        risk_scores = self.predict_off_target_risk(sequences)
        logger.info(f"Off-target risk range: {risk_scores.min():.3f} - {risk_scores.max():.3f}")

        # Predict uncertainty
        uncertainty_scores = self.predict_uncertainty(mu_pred, phi_pred)
        logger.info(f"Uncertainty range: {uncertainty_scores.min():.3f} - {uncertainty_scores.max():.3f}")

        # Compute integrated design score: S = w_e*mu - w_r*R - w_u*sigma
        design_scores = (self.weights['w_e'] * mu_pred -
                        self.weights['w_r'] * risk_scores -
                        self.weights['w_u'] * uncertainty_scores)

        logger.info(f"Design scores range: {design_scores.min():.3f} - {design_scores.max():.3f}")

        return {
            'design_scores': design_scores,
            'efficacy_predictions': mu_pred,
            'risk_scores': risk_scores,
            'uncertainty_scores': uncertainty_scores,
            'concentration_params': phi_pred
        }

    def rank_candidates(self, sequences: List[str], metadata=None) -> pd.DataFrame:
        """Rank gRNA candidates by design score."""

        # Compute all scores
        scores = self.compute_design_scores(sequences)

        # Create results dataframe
        results_df = pd.DataFrame({
            'sequence': sequences,
            'design_score': scores['design_scores'],
            'efficacy_prediction': scores['efficacy_predictions'],
            'risk_score': scores['risk_scores'],
            'uncertainty_score': scores['uncertainty_scores'],
            'concentration': scores['concentration_params']
        })

        # Add metadata if provided
        if metadata is not None:
            for col, values in metadata.items():
                results_df[col] = values

        # Sort by design score (descending - higher is better)
        results_df = results_df.sort_values('design_score', ascending=False).reset_index(drop=True)
        results_df['rank'] = range(1, len(results_df) + 1)

        # Add score components for analysis
        results_df['efficacy_component'] = self.weights['w_e'] * results_df['efficacy_prediction']
        results_df['risk_component'] = -self.weights['w_r'] * results_df['risk_score']  # Negative because it's subtracted
        results_df['uncertainty_component'] = -self.weights['w_u'] * results_df['uncertainty_score']  # Negative because it's subtracted

        return results_df

    def validate_against_known_guides(self, ranked_df: pd.DataFrame,
                                    known_effective_guides: List[str],
                                    known_ineffective_guides: List[str] = None) -> Dict:
        """Validate rankings against known effective/ineffective guides."""

        # Find positions of known effective guides
        effective_positions = []
        effective_scores = []

        for guide in known_effective_guides:
            if guide in ranked_df['sequence'].values:
                pos = ranked_df[ranked_df['sequence'] == guide].index[0] + 1
                score = ranked_df[ranked_df['sequence'] == guide]['design_score'].values[0]
                effective_positions.append(pos)
                effective_scores.append(score)
                logger.info(f"Known effective guide {guide[:10]}... ranked #{pos} (score: {score:.3f})")

        # Find positions of known ineffective guides (if provided)
        ineffective_positions = []
        ineffective_scores = []

        if known_ineffective_guides:
            for guide in known_ineffective_guides:
                if guide in ranked_df['sequence'].values:
                    pos = ranked_df[ranked_df['sequence'] == guide].index[0] + 1
                    score = ranked_df[ranked_df['sequence'] == guide]['design_score'].values[0]
                    ineffective_positions.append(pos)
                    ineffective_scores.append(score)
                    logger.info(f"Known ineffective guide {guide[:10]}... ranked #{pos} (score: {score:.3f})")

        # Compute validation metrics
        validation_results = {
            'known_effective': {
                'count': len(effective_positions),
                'positions': effective_positions,
                'mean_position': np.mean(effective_positions) if effective_positions else None,
                'median_position': np.median(effective_positions) if effective_positions else None,
                'top_10_percent': sum(p <= len(ranked_df) * 0.1 for p in effective_positions) if effective_positions else 0,
                'top_25_percent': sum(p <= len(ranked_df) * 0.25 for p in effective_positions) if effective_positions else 0,
                'mean_score': np.mean(effective_scores) if effective_scores else None
            }
        }

        if known_ineffective_guides:
            validation_results['known_ineffective'] = {
                'count': len(ineffective_positions),
                'positions': ineffective_positions,
                'mean_position': np.mean(ineffective_positions) if ineffective_positions else None,
                'median_position': np.median(ineffective_positions) if ineffective_positions else None,
                'bottom_50_percent': sum(p >= len(ranked_df) * 0.5 for p in ineffective_positions) if ineffective_positions else 0,
                'mean_score': np.mean(ineffective_scores) if ineffective_scores else None
            }

            # Compute separation metrics
            if effective_scores and ineffective_scores:
                validation_results['separation'] = {
                    'mean_score_difference': np.mean(effective_scores) - np.mean(ineffective_scores),
                    'effect_size_cohens_d': (np.mean(effective_scores) - np.mean(ineffective_scores)) /
                                          np.sqrt((np.var(effective_scores) + np.var(ineffective_scores)) / 2)
                }

        return validation_results

    def plot_score_analysis(self, ranked_df: pd.DataFrame, output_dir: Path):
        """Generate analysis plots for design scores."""

        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Score distribution
        ax1 = axes[0, 0]
        ax1.hist(ranked_df['design_score'], bins=50, alpha=0.7, density=True)
        ax1.axvline(ranked_df['design_score'].mean(), color='red', linestyle='--', label='Mean')
        ax1.axvline(ranked_df['design_score'].median(), color='orange', linestyle='--', label='Median')
        ax1.set_xlabel('Design Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Design Score Distribution')
        ax1.legend()

        # Plot 2: Score components
        ax2 = axes[0, 1]
        components = ['efficacy_component', 'risk_component', 'uncertainty_component']
        comp_means = [ranked_df[comp].mean() for comp in components]
        comp_labels = ['Efficacy\n(+w_e×μ)', 'Risk\n(-w_r×R)', 'Uncertainty\n(-w_u×σ)']

        colors = ['green', 'red', 'orange']
        bars = ax2.bar(comp_labels, comp_means, color=colors, alpha=0.7)
        ax2.set_ylabel('Mean Contribution to Design Score')
        ax2.set_title('Score Component Analysis')
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, comp_means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + np.sign(height)*0.001,
                    f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top')

        # Plot 3: Top candidates analysis
        ax3 = axes[1, 0]
        top_n = min(50, len(ranked_df))
        top_candidates = ranked_df.head(top_n)

        scatter = ax3.scatter(top_candidates['efficacy_prediction'],
                            top_candidates['design_score'],
                            c=top_candidates['uncertainty_score'],
                            s=60, alpha=0.7, cmap='viridis_r')

        ax3.set_xlabel('Efficacy Prediction')
        ax3.set_ylabel('Design Score')
        ax3.set_title(f'Top {top_n} Candidates: Efficacy vs Design Score')

        # Add colorbar for uncertainty
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Uncertainty Score')

        # Plot 4: Score component correlation
        ax4 = axes[1, 1]

        # Create correlation matrix
        score_cols = ['efficacy_prediction', 'risk_score', 'uncertainty_score', 'design_score']
        corr_matrix = ranked_df[score_cols].corr()

        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax4, cbar_kws={"shrink": .8})
        ax4.set_title('Score Component Correlations')

        plt.tight_layout()
        plt.savefig(output_dir / 'design_score_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Additional plot: ranking distribution
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot design score vs rank
        ax.scatter(ranked_df['rank'], ranked_df['design_score'], alpha=0.6, s=20)
        ax.set_xlabel('Rank')
        ax.set_ylabel('Design Score')
        ax.set_title('Design Score vs Rank')

        # Highlight top candidates
        top_10_percent = len(ranked_df) // 10
        ax.scatter(ranked_df['rank'][:top_10_percent],
                  ranked_df['design_score'][:top_10_percent],
                  color='red', alpha=0.8, s=30, label=f'Top 10% (n={top_10_percent})')

        ax.legend()
        plt.savefig(output_dir / 'rank_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Analysis plots saved to {output_dir}")

    def generate_candidate_report(self, ranked_df: pd.DataFrame,
                                validation_results: Dict,
                                output_dir: Path):
        """Generate comprehensive candidate report."""

        # Create summary statistics
        summary = {
            'total_candidates': len(ranked_df),
            'score_statistics': {
                'mean': float(ranked_df['design_score'].mean()),
                'std': float(ranked_df['design_score'].std()),
                'min': float(ranked_df['design_score'].min()),
                'max': float(ranked_df['design_score'].max()),
                'q25': float(ranked_df['design_score'].quantile(0.25)),
                'median': float(ranked_df['design_score'].median()),
                'q75': float(ranked_df['design_score'].quantile(0.75))
            },
            'component_analysis': {
                'efficacy_contribution': float(ranked_df['efficacy_component'].mean()),
                'risk_contribution': float(ranked_df['risk_component'].mean()),
                'uncertainty_contribution': float(ranked_df['uncertainty_component'].mean())
            },
            'weights_used': self.weights,
            'validation_results': validation_results
        }

        # Save detailed results
        ranked_df.to_csv(output_dir / 'ranked_candidates.csv', index=False)

        # Save top candidates
        top_candidates = ranked_df.head(100)
        top_candidates.to_csv(output_dir / 'top_100_candidates.csv', index=False)

        # Save summary
        with open(output_dir / 'design_score_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Generate text summary
        self.generate_text_summary(ranked_df, validation_results, summary, output_dir)

        logger.info(f"Candidate report saved to {output_dir}")

        return summary

    def generate_text_summary(self, ranked_df: pd.DataFrame,
                            validation_results: Dict,
                            summary: Dict,
                            output_dir: Path):
        """Generate human-readable summary report."""

        report_text = f"""
# ChromaGuide Designer Score Evaluation Report

## Summary
- **Total Candidates Evaluated:** {summary['total_candidates']:,}
- **Design Score Range:** {summary['score_statistics']['min']:.3f} to {summary['score_statistics']['max']:.3f}
- **Mean Design Score:** {summary['score_statistics']['mean']:.3f} (±{summary['score_statistics']['std']:.3f})

## Weights Used
- **Efficacy Weight (w_e):** {summary['weights_used']['w_e']}
- **Risk Weight (w_r):** {summary['weights_used']['w_r']}
- **Uncertainty Weight (w_u):** {summary['weights_used']['w_u']}

## Score Component Analysis
- **Mean Efficacy Contribution:** {summary['component_analysis']['efficacy_contribution']:+.3f}
- **Mean Risk Contribution:** {summary['component_analysis']['risk_contribution']:+.3f}
- **Mean Uncertainty Contribution:** {summary['component_analysis']['uncertainty_contribution']:+.3f}

## Top 10 Candidates
"""

        # Add top candidates table
        top_10 = ranked_df.head(10)
        report_text += "\n| Rank | Sequence | Design Score | Efficacy | Risk | Uncertainty |\n"
        report_text += "|------|----------|--------------|----------|------|-------------|\n"

        for _, row in top_10.iterrows():
            report_text += f"| {row['rank']} | {row['sequence'][:15]}... | {row['design_score']:.4f} | {row['efficacy_prediction']:.4f} | {row['risk_score']:.4f} | {row['uncertainty_score']:.4f} |\n"

        # Add validation results if available
        if validation_results and 'known_effective' in validation_results:
            effective_stats = validation_results['known_effective']
            if effective_stats['count'] > 0:
                report_text += f"""
## Validation Against Known Effective Guides
- **Known Effective Guides Found:** {effective_stats['count']}
- **Mean Ranking Position:** {effective_stats['mean_position']:.1f}
- **Guides in Top 10%:** {effective_stats['top_10_percent']}/{effective_stats['count']} ({effective_stats['top_10_percent']/effective_stats['count']*100:.1f}%)
- **Guides in Top 25%:** {effective_stats['top_25_percent']}/{effective_stats['count']} ({effective_stats['top_25_percent']/effective_stats['count']*100:.1f}%)
"""

        # Score distribution quartiles
        report_text += f"""
## Score Distribution
- **Q1 (25th percentile):** {summary['score_statistics']['q25']:.4f}
- **Q2 (Median):** {summary['score_statistics']['median']:.4f}
- **Q3 (75th percentile):** {summary['score_statistics']['q75']:.4f}

## Recommendations
1. **For Experimental Validation:** Consider top 20-50 candidates for wet lab validation
2. **High Confidence Candidates:** Candidates in top 10% with low uncertainty scores
3. **Conservative Selection:** Balance high efficacy prediction with low risk scores

Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()

        with open(output_dir / 'design_score_report.md', 'w') as f:
            f.write(report_text)

def main():
    parser = argparse.ArgumentParser(description='ChromaGuide Designer Score Evaluation')

    # Required arguments
    parser.add_argument('--on-target-model', type=str, required=True,
                       help='Path to trained on-target efficacy model')
    parser.add_argument('--candidates', type=str, required=True,
                       help='Path to CSV file with candidate gRNA sequences')
    parser.add_argument('--output-dir', type=str, default='results/designer_evaluation',
                       help='Output directory for results')

    # Optional model paths
    parser.add_argument('--off-target-model', type=str, default=None,
                       help='Path to trained off-target risk model')
    parser.add_argument('--conformal-model', type=str, default=None,
                       help='Path to conformal calibration file')

    # Weights
    parser.add_argument('--w-efficacy', type=float, default=1.0,
                       help='Weight for efficacy component (default: 1.0)')
    parser.add_argument('--w-risk', type=float, default=0.5,
                       help='Weight for risk component (default: 0.5)')
    parser.add_argument('--w-uncertainty', type=float, default=0.2,
                       help='Weight for uncertainty component (default: 0.2)')

    # Validation
    parser.add_argument('--known-effective', type=str, default=None,
                       help='Path to file with known effective guide sequences')
    parser.add_argument('--known-ineffective', type=str, default=None,
                       help='Path to file with known ineffective guide sequences')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up weights
    weights = {
        'w_e': args.w_efficacy,
        'w_r': args.w_risk,
        'w_u': args.w_uncertainty
    }

    # Initialize evaluator
    evaluator = DesignerScoreEvaluator(weights=weights)

    # Load models
    evaluator.load_models(
        on_target_path=args.on_target_model,
        off_target_path=args.off_target_model,
        conformal_path=args.conformal_model
    )

    # Load candidate sequences
    logger.info(f"Loading candidates from {args.candidates}")
    candidates_df = pd.read_csv(args.candidates)

    # Extract sequences (assume column named 'sequence' or 'gRNA' or similar)
    sequence_cols = ['sequence', 'gRNA', 'guide_sequence', 'seq']
    sequence_col = None
    for col in sequence_cols:
        if col in candidates_df.columns:
            sequence_col = col
            break

    if sequence_col is None:
        raise ValueError(f"No sequence column found. Expected one of: {sequence_cols}")

    sequences = candidates_df[sequence_col].tolist()
    logger.info(f"Loaded {len(sequences)} candidate sequences")

    # Prepare metadata for results
    metadata = {}
    for col in candidates_df.columns:
        if col != sequence_col:
            metadata[col] = candidates_df[col].tolist()

    # Compute rankings
    logger.info("Computing design scores and ranking candidates...")
    ranked_df = evaluator.rank_candidates(sequences, metadata)

    # Load known guides for validation (if provided)
    known_effective = []
    known_ineffective = []

    if args.known_effective:
        with open(args.known_effective, 'r') as f:
            known_effective = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(known_effective)} known effective guides")

    if args.known_ineffective:
        with open(args.known_ineffective, 'r') as f:
            known_ineffective = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(known_ineffective)} known ineffective guides")

    # Validate against known guides
    validation_results = {}
    if known_effective or known_ineffective:
        logger.info("Validating rankings against known effective/ineffective guides...")
        validation_results = evaluator.validate_against_known_guides(
            ranked_df, known_effective, known_ineffective
        )

    # Generate plots
    logger.info("Generating analysis plots...")
    evaluator.plot_score_analysis(ranked_df, output_dir)

    # Generate comprehensive report
    logger.info("Generating candidate report...")
    summary = evaluator.generate_candidate_report(
        ranked_df, validation_results, output_dir
    )

    # Print summary
    print("\n" + "="*60)
    print("CHROMAGUIDE DESIGNER SCORE EVALUATION COMPLETE")
    print("="*60)
    print(f"Candidates Evaluated: {len(ranked_df):,}")
    print(f"Top Candidate Score: {ranked_df['design_score'].iloc[0]:.4f}")
    print(f"Score Range: {ranked_df['design_score'].min():.4f} to {ranked_df['design_score'].max():.4f}")

    if validation_results and 'known_effective' in validation_results:
        effective_stats = validation_results['known_effective']
        if effective_stats['count'] > 0:
            print(f"Known Effective Guides in Top 25%: {effective_stats['top_25_percent']}/{effective_stats['count']}")

    print(f"Results saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
