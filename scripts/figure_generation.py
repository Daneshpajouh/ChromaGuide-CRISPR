"""
FIGURE GENERATION
Publication-quality visualizations for PhD thesis
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, gaussian_kde
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class FigureGenerator:
    """Generate publication-quality figures."""
    
    def __init__(self, results_dir: Path, output_dir: Path):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.load_all_results()
    
    def load_all_results(self):
        """Load all model results."""
        result_files = self.results_dir.glob('*/results.json')
        
        for result_file in result_files:
            model_name = result_file.parent.name
            
            try:
                with open(result_file) as f:
                    data = json.load(f)
                
                self.models[model_name] = {
                    'predictions': np.array(data['predictions']),
                    'labels': np.array(data['labels'])
                }
                
                logger.info(f"✓ Loaded {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
    
    def scatter_plot_predictions(self):
        """Scatter plots: predicted vs actual for each model."""
        logger.info("Generating scatter plots...")
        
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('ChromaGuide Predictions vs Ground Truth', fontsize=16, fontweight='bold')
        
        for idx, (model_name, data) in enumerate(self.models.items()):
            ax = axes[idx // 2, idx % 2]
            
            preds = data['predictions']
            labels = data['labels']
            rho, pval = spearmanr(preds, labels)
            
            # Scatter plot with density coloring
            try:
                xy = np.vstack([preds, labels])
                z = gaussian_kde(xy)(xy)
                scatter = ax.scatter(preds, labels, c=z, s=50, alpha=0.6, cmap='viridis')
                plt.colorbar(scatter, ax=ax, label='Density')
            except:
                ax.scatter(preds, labels, alpha=0.5, s=30)
            
            # Perfect prediction line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect prediction')
            
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal')
            ax.set_xlabel('Predicted Efficacy', fontweight='bold')
            ax.set_ylabel('Ground Truth', fontweight='bold')
            ax.set_title(f'{model_name}\nρ={rho:.4f} (p={pval:.2e})', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide empty subplot
        if n_models < 4:
            axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scatter_predictions.pdf', dpi=300, bbox_inches='tight')
        logger.info("✓ Saved scatter plots")
        plt.close()
    
    def model_comparison_bars(self):
        """Bar plot comparing all models by Spearman correlation."""
        logger.info("Generating model comparison plot...")
        
        model_names = []
        spearman_rhos = []
        
        for model_name, data in self.models.items():
            rho, _ = spearmanr(data['predictions'], data['labels'])
            model_names.append(model_name)
            spearman_rhos.append(rho)
        
        # Sort by performance
        sorted_idx = np.argsort(spearman_rhos)[::-1]
        model_names = [model_names[i] for i in sorted_idx]
        spearman_rhos = [spearman_rhos[i] for i in sorted_idx]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2ecc71' if i == 0 else '#3498db' if 'chromaguide' in model_names[i] 
                  else '#e74c3c' for i in range(len(model_names))]
        
        bars = ax.barh(range(len(model_names)), spearman_rhos, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names, fontweight='bold')
        ax.set_xlabel('Spearman Correlation', fontweight='bold', fontsize=12)
        ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, spearman_rhos)):
            ax.text(value + 0.01, i, f'{value:.4f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.pdf', dpi=300, bbox_inches='tight')
        logger.info("✓ Saved model comparison")
        plt.close()
    
    def residual_plots(self):
        """Residual analysis plots."""
        logger.info("Generating residual plots...")
        
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle('Residual Analysis', fontsize=14, fontweight='bold')
        
        for idx, (model_name, data) in enumerate(self.models.items()):
            ax = axes[idx]
            
            preds = data['predictions']
            labels = data['labels']
            residuals = preds - labels
            
            ax.scatter(labels, residuals, alpha=0.5, s=30)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax.set_xlabel('Ground Truth', fontweight='bold')
            ax.set_ylabel('Residuals', fontweight='bold')
            ax.set_title(model_name, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add mean line
            ax.axhline(y=np.mean(residuals), color='g', linestyle=':', alpha=0.7, label=f'Mean={np.mean(residuals):.3f}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'residuals.pdf', dpi=300, bbox_inches='tight')
        logger.info("✓ Saved residual plots")
        plt.close()
    
    def error_distribution(self):
        """Distribution of absolute errors."""
        logger.info("Generating error distribution...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name, data in self.models.items():
            errors = np.abs(data['predictions'] - data['labels'])
            ax.hist(errors, bins=30, alpha=0.6, label=model_name, edgecolor='black')
        
        ax.set_xlabel('Absolute Error', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Prediction Error Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_distribution.pdf', dpi=300, bbox_inches='tight')
        logger.info("✓ Saved error distribution")
        plt.close()
    
    def calibration_plot(self):
        """Conformal prediction calibration plot."""
        logger.info("Generating calibration plot...")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        confidences = np.linspace(0.5, 1.0, 20)
        
        for model_name, data in self.models.items():
            preds = data['predictions']
            labels = data['labels']
            
            empirical_coverages = []
            for conf in confidences:
                residuals = np.abs(preds - labels)
                threshold = np.percentile(residuals, conf * 100)
                coverage = np.mean(residuals <= threshold)
                empirical_coverages.append(coverage)
            
            ax.plot(confidences, empirical_coverages, marker='o', label=model_name, linewidth=2)
        
        # Perfect calibration line
        ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', alpha=0.5, label='Perfect calibration')
        
        ax.set_xlabel('Nominal Confidence', fontweight='bold')
        ax.set_ylabel('Empirical Coverage', fontweight='bold')
        ax.set_title('Conformal Prediction Calibration', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.5, 1.0])
        ax.set_ylim([0.5, 1.05])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'calibration.pdf', dpi=300, bbox_inches='tight')
        logger.info("✓ Saved calibration plot")
        plt.close()
    
    def ranking_consistency(self):
        """Ranking consistency across models."""
        logger.info("Generating ranking consistency plot...")
        
        if len(self.models) < 2:
            logger.warning("Need at least 2 models for ranking consistency")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get predictions from baseline and best model
        baseline_name = 'seq_only_baseline'
        best_name = 'chromaguide_full'
        
        if baseline_name in self.models and best_name in self.models:
            baseline_preds = self.models[baseline_name]['predictions']
            best_preds = self.models[best_name]['predictions']
            labels = self.models[best_name]['labels']
            
            # Rank by baseline
            baseline_rank = np.argsort(np.abs(baseline_preds - labels))
            # Rank by best
            best_rank = np.argsort(np.abs(best_preds - labels))
            
            # Spearman correlation of rankings
            rank_corr, rank_pval = spearmanr(baseline_rank, best_rank)
            
            ax.scatter(baseline_rank, best_rank, alpha=0.5, s=30)
            ax.set_xlabel(f'{baseline_name} ranking', fontweight='bold')
            ax.set_ylabel(f'{best_name} ranking', fontweight='bold')
            ax.set_title(f'Ranking Consistency (ρ={rank_corr:.3f}, p={rank_pval:.2e})', 
                        fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ranking_consistency.pdf', dpi=300, bbox_inches='tight')
        logger.info("✓ Saved ranking consistency plot")
        plt.close()
    
    def generate_all_figures(self):
        """Generate all figures."""
        logger.info("="*60)
        logger.info("GENERATING PUBLICATION-QUALITY FIGURES")
        logger.info("="*60)
        
        self.scatter_plot_predictions()
        self.model_comparison_bars()
        self.residual_plots()
        self.error_distribution()
        self.calibration_plot()
        self.ranking_consistency()
        
        logger.info("\n" + "="*60)
        logger.info(f"✓ ALL FIGURES SAVED TO {self.output_dir}")
        logger.info("="*60)


def main():
    """Main figure generation."""
    results_dir = Path("/project/def-bengioy/chromaguide_results")
    figure_dir = results_dir / "figures"
    
    generator = FigureGenerator(results_dir, figure_dir)
    generator.generate_all_figures()


if __name__ == "__main__":
    main()
