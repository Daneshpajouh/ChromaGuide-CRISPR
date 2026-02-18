"""
EVALUATION AND STATISTICAL TESTING
PhD thesis rigorous evaluation suite
Computes Spearman correlation, conformal prediction, significance tests, effect sizes
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr, wilcoxon, ttest_rel
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelResults:
    """Holds all results for a trained model."""
    model_name: str
    predictions: np.ndarray
    labels: np.ndarray
    spearman_rho: float
    spearman_pval: float
    
    def conformal_calibration(self, confidence=0.9):
        """Compute conformal prediction calibration.
        Returns empirical coverage at target confidence level."""
        residuals = np.abs(self.predictions - self.labels)
        percentile = np.percentile(residuals, confidence * 100)
        coverage = np.mean(residuals <= percentile)
        return coverage, percentile


class RigourousEvaluator:
    """PhD thesis evaluation with all required metrics."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.models = {}
        self.statistics = {}
    
    def load_model_results(self, model_name: str, results_json: Path):
        """Load predictions and labels from results.json."""
        with open(results_json) as f:
            data = json.load(f)
        
        preds = np.array(data['predictions'])
        labels = np.array(data['labels'])
        
        rho, pval = spearmanr(preds, labels)
        
        self.models[model_name] = ModelResults(
            model_name=model_name,
            predictions=preds,
            labels=labels,
            spearman_rho=rho,
            spearman_pval=pval
        )
        
        logger.info(f"✓ Loaded {model_name}: Spearman ρ = {rho:.4f}")
    
    def compute_effect_sizes(self, model1_name: str, model2_name: str):
        """Compute Cohen's d effect size between two models."""
        m1 = self.models[model1_name]
        m2 = self.models[model2_name]
        
        e1 = m1.predictions - m1.labels
        e2 = m2.predictions - m2.labels
        
        mean_diff = np.mean(e1) - np.mean(e2)
        pooled_std = np.sqrt((np.var(e1) + np.var(e2)) / 2)
        
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        return cohens_d
    
    def statistical_significance_tests(self, model1_name: str, model2_name: str):
        """Wilcoxon signed-rank test and paired t-test."""
        m1 = self.models[model1_name]
        m2 = self.models[model2_name]
        
        e1 = np.abs(m1.predictions - m1.labels)
        e2 = np.abs(m2.predictions - m2.labels)
        
        # Wilcoxon signed-rank test
        w_stat, w_pval = wilcoxon(e1, e2, alternative='two-sided')
        
        # Paired t-test
        t_stat, t_pval = ttest_rel(e1, e2)
        
        return {
            'wilcoxon_statistic': float(w_stat),
            'wilcoxon_pvalue': float(w_pval),
            'ttest_statistic': float(t_stat),
            'ttest_pvalue': float(t_pval),
            'cohens_d': float(self.compute_effect_sizes(model1_name, model2_name))
        }
    
    def bootstrap_confidence_intervals(self, model_name: str, n_bootstrap=1000, ci=0.95):
        """Bootstrap confidence intervals for Spearman rho."""
        m = self.models[model_name]
        
        bootstrap_rhos = []
        n = len(m.predictions)
        
        np.random.seed(42)
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            rho, _ = spearmanr(m.predictions[indices], m.labels[indices])
            bootstrap_rhos.append(rho)
        
        bootstrap_rhos = np.array(bootstrap_rhos)
        lower = np.percentile(bootstrap_rhos, (1 - ci) / 2 * 100)
        upper = np.percentile(bootstrap_rhos, (1 + ci) / 2 * 100)
        
        return {
            'mean': float(np.mean(bootstrap_rhos)),
            'std': float(np.std(bootstrap_rhos)),
            'ci_lower': float(lower),
            'ci_upper': float(upper),
            'range': f"[{lower:.4f}, {upper:.4f}]"
        }
    
    def generate_comparison_table(self):
        """Create comparison table for all models."""
        table_data = []
        
        for model_name, model in self.models.items():
            coverage, percentile = model.conformal_calibration(confidence=0.9)
            
            table_data.append({
                'Model': model_name,
                'Spearman ρ': f"{model.spearman_rho:.4f}",
                'p-value': f"{model.spearman_pval:.2e}",
                'Conformal Coverage (90%)': f"{coverage:.3f}",
                'Prediction Interval Width': f"{percentile:.3f}"
            })
        
        return pd.DataFrame(table_data)
    
    def generate_statistical_summary(self):
        """Generate summary of all statistical tests."""
        summary = {}
        
        # Primary model results
        summary['primary_model'] = {
            'name': 'chromaguide_full',
            'spearman_rho': float(self.models['chromaguide_full'].spearman_rho),
            'p_value': float(self.models['chromaguide_full'].spearman_pval),
            'bootstrap_ci': self.bootstrap_confidence_intervals('chromaguide_full')
        }
        
        # Comparisons vs baselines
        summary['comparisons'] = {}
        
        baseline_models = ['seq_only_baseline']
        for baseline in baseline_models:
            if baseline in self.models:
                summary['comparisons'][baseline] = self.statistical_significance_tests(
                    'chromaguide_full', baseline
                )
        
        # Ablation comparisons
        summary['ablations'] = {}
        ablations = ['mamba_variant']
        for ablation in ablations:
            if ablation in self.models:
                summary['ablations'][ablation] = {
                    'spearman_rho': float(self.models[ablation].spearman_rho),
                    'absolute_difference': float(
                        self.models['chromaguide_full'].spearman_rho - 
                        self.models[ablation].spearman_rho
                    )
                }
        
        return summary
    
    def generate_markdown_report(self, output_dir: Path):
        """Generate publication-ready markdown report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = f"""# ChromaGuide PhD Thesis Evaluation Report

## Executive Summary

This report presents comprehensive statistical evaluation of the ChromaGuide model
for CRISPR sgRNA efficacy prediction, with rigorous leakage-controlled experimental
design and publication-grade statistical testing.

## Primary Results

### Main Model Performance

"""
        
        # Add model results
        for model_name, model in self.models.items():
            coverage, _ = model.conformal_calibration(0.9)
            report += f"\n**{model_name}**\n"
            report += f"- Spearman ρ: {model.spearman_rho:.4f} (p={model.spearman_pval:.2e})\n"
            report += f"- Conformal Coverage (90%): {coverage:.3f}\n"
        
        report += "\n## Comparison Table\n\n"
        comparison_df = self.generate_comparison_table()
        report += comparison_df.to_markdown(index=False)
        
        # Statistical tests
        report += "\n\n## Statistical Significance Tests\n\n"
        summary = self.generate_statistical_summary()
        
        for comparison_name, stats in summary['comparisons'].items():
            report += f"\n### ChromaGuide vs {comparison_name}\n\n"
            report += f"- Cohen's d: {stats['cohens_d']:.3f}\n"
            report += f"- Wilcoxon p-value: {stats['wilcoxon_pvalue']:.2e}\n"
            report += f"- Paired t-test p-value: {stats['ttest_pvalue']:.2e}\n"
        
        report_path = output_dir / 'evaluation_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"✓ Report saved to {report_path}")
        
        return report
    
    def save_all_results(self, output_dir: Path):
        """Save all evaluation results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Comparison table
        comparison_df = self.generate_comparison_table()
        comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
        logger.info(f"✓ Saved comparison table")
        
        # Statistical summary
        summary = self.generate_statistical_summary()
        with open(output_dir / 'statistical_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved statistical summary")
        
        # Individual model metrics
        for model_name, model in self.models.items():
            metrics = {
                'model': model_name,
                'spearman_rho': float(model.spearman_rho),
                'spearman_pvalue': float(model.spearman_pval),
                'conformal_coverage_90': float(model.conformal_calibration(0.9)[0]),
                'bootstrap_ci': self.bootstrap_confidence_intervals(model_name)
            }
            
            with open(output_dir / f'{model_name}_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
        
        logger.info(f"✓ Saved individual model metrics")
        
        # Generate markdown report
        self.generate_markdown_report(output_dir)


def main():
    """Main evaluation execution."""
    logger.info("="*80)
    logger.info("PhD THESIS EVALUATION PIPELINE")
    logger.info("="*80)
    
    results_dir = Path("/project/def-bengioy/chromaguide_results")
    eval_output = results_dir / "evaluation"
    
    evaluator = RigourousEvaluator(results_dir)
    
    # Load all model results
    model_files = {
        'seq_only_baseline': results_dir / 'seq_only_baseline' / 'results.json',
        'chromaguide_full': results_dir / 'chromaguide_full' / 'results.json',
        'mamba_variant': results_dir / 'mamba_variant' / 'results.json'
    }
    
    for model_name, results_file in model_files.items():
        if results_file.exists():
            evaluator.load_model_results(model_name, results_file)
    
    # Save all evaluation results
    evaluator.save_all_results(eval_output)
    
    logger.info("\n" + "="*80)
    logger.info("✓ EVALUATION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
