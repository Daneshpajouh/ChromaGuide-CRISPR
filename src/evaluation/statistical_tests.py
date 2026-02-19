#!/usr/bin/env python3
"""
Statistical Testing Framework for CRISPR Prediction Evaluation.

Implements rigorous statistical validation required for scientific publication:
1. Wilcoxon signed-rank test (non-parametric, paired data)
2. Paired t-test (parametric alternative)
3. Cohen's d effect size (standardized difference)
4. Bootstrap confidence intervals for correlation metrics
5. Multiple comparison correction (Bonferroni, Benjamini-Hochberg)

Ensures p < 0.001 significance threshold as specified in proposal.

References:
  - Wilcoxon (1945): Individual comparisons by ranking methods
  - Cohen (1988): Statistical power analysis for behavioral sciences
  - Benjamini & Hochberg (1995): Controlling false discovery rate
  - Davison & Hinkley (1997): Bootstrap methods and their applications
"""

import numpy as np
from scipy import stats
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import warnings


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    significant_at_001: bool  # p < 0.001
    significant_at_005: bool  # p < 0.05
    effect_size: float  # Cohen's d or similar
    ci_lower: float  # 95% confidence interval lower
    ci_upper: float  # 95% confidence interval upper
    interpretation: str

    def __str__(self) -> str:
        """Pretty print result."""
        sig_marker = "***" if self.significant_at_001 else ("**" if self.significant_at_005 else "")
        return (
            f"{self.test_name}: "
            f"p={self.p_value:.6f}{sig_marker}, "
            f"d={self.effect_size:.4f}, "
            f"CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        )


class StatisticalTester:
    """
    Comprehensive statistical testing suite for model evaluation.
    """

    @staticmethod
    def wilcoxon_signed_rank(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alternative: str = 'two-sided'
    ) -> StatisticalTestResult:
        """
        Wilcoxon signed-rank test for paired samples.

        Tests whether predictions differ significantly from true values.
        Non-parametric (no normality assumption required).

        H0: Medians are equal (model predictions match true values)
        H1: Medians differ

        Args:
            y_true: True target values
            y_pred: Predicted values
            alternative: 'two-sided' (default), 'less', 'greater'

        Returns:
            StatisticalTestResult with p-value and effect size
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        # Calculate signed differences
        differences = y_pred - y_true

        # Remove zeros (tied pairs)
        nonzero_diff = differences[differences != 0]

        if len(nonzero_diff) < 3:
            warnings.warn("< 3 non-zero differences, test may be unreliable")

        # Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(nonzero_diff, alternative=alternative)

        # Effect size: r = Z / sqrt(N)
        n = len(nonzero_diff)
        z_score = abs(stats.norm.ppf(p_value / 2)) if p_value > 0 else 0
        r_effect = z_score / np.sqrt(n) if n > 0 else 0

        # Bootstrap CI for median difference
        ci_lower, ci_upper = StatisticalTester.bootstrap_ci(
            differences[differences != 0], statistic=np.median
        )

        # Interpretation
        if p_value < 0.001:
            interpretation = "Highly significant difference (p < 0.001)"
        elif p_value < 0.05:
            interpretation = "Significant difference (p < 0.05)"
        else:
            interpretation = "No significant difference (p >= 0.05)"

        return StatisticalTestResult(
            test_name="Wilcoxon Signed-Rank",
            statistic=float(statistic),
            p_value=float(p_value),
            significant_at_001=p_value < 0.001,
            significant_at_005=p_value < 0.05,
            effect_size=float(r_effect),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            interpretation=interpretation
        )

    @staticmethod
    def paired_t_test(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alternative: str = 'two-sided'
    ) -> StatisticalTestResult:
        """
        Paired t-test (parametric alternative to Wilcoxon).

        Assumes normally distributed differences.
        More powerful if normality holds; less robust otherwise.

        Args:
            y_true: True values
            y_pred: Predicted values
            alternative: 'two-sided', 'less', 'greater'

        Returns:
            StatisticalTestResult
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        differences = y_pred - y_true

        # Paired t-test
        t_stat, p_value = stats.ttest_1samp(differences, 0, alternative=alternative)

        # Cohen's d for paired samples
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohen_d = mean_diff / std_diff if std_diff > 0 else 0

        # Confidence interval on mean difference
        n = len(differences)
        se = std_diff / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, n - 1)  # 95% CI
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se

        # Interpretation
        if p_value < 0.001:
            interpretation = "Highly significant difference (p < 0.001)"
        elif p_value < 0.05:
            interpretation = "Significant difference (p < 0.05)"
        else:
            interpretation = "No significant difference"

        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=float(t_stat),
            p_value=float(p_value),
            significant_at_001=p_value < 0.001,
            significant_at_005=p_value < 0.05,
            effect_size=float(cohen_d),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            interpretation=interpretation
        )

    @staticmethod
    def correlation_significance(
        x: np.ndarray,
        y: np.ndarray,
        method: str = 'spearman'
    ) -> StatisticalTestResult:
        """
        Test correlation coefficient for significance.

        Args:
            x: First variable
            y: Second variable
            method: 'spearman' (rank correlation) or 'pearson' (linear)

        Returns:
            StatisticalTestResult
        """
        if method == 'spearman':
            correlation, p_value = stats.spearmanr(x, y)
            test_name = "Spearman's Rho"
        elif method == 'pearson':
            correlation, p_value = stats.pearsonr(x, y)
            test_name = "Pearson's r"
        else:
            raise ValueError(f"Unknown method: {method}")

        # Fisher's z-transformation for CI
        z = np.arctanh(correlation)
        se = 1 / np.sqrt(len(x) - 3)
        z_crit = stats.norm.ppf(0.975)
        ci_lower = np.tanh(z - z_crit * se)
        ci_upper = np.tanh(z + z_crit * se)

        # Effect size: r = correlation
        effect_size = correlation

        # Interpretation
        if abs(correlation) < 0.3:
            strength = "weak"
        elif abs(correlation) < 0.7:
            strength = "moderate"
        else:
            strength = "strong"

        if p_value < 0.001:
            interpretation = f"Strong significance: {strength} {test_name}"
        else:
            interpretation = f"Not significant: {strength} {test_name}"

        return StatisticalTestResult(
            test_name=test_name,
            statistic=float(correlation),
            p_value=float(p_value),
            significant_at_001=p_value < 0.001,
            significant_at_005=p_value < 0.05,
            effect_size=float(effect_size),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            interpretation=interpretation
        )

    @staticmethod
    def cohens_d(
        group1: np.ndarray,
        group2: np.ndarray,
        pooled: bool = True
    ) -> float:
        """
        Compute Cohen's d effect size.

        d = (μ1 - μ2) / σ_pooled

        Interpretation:
        - |d| < 0.2: negligible
        - 0.2 ≤ |d| < 0.5: small
        - 0.5 ≤ |d| < 0.8: medium
        - |d| ≥ 0.8: large

        Args:
            group1: First group scores
            group2: Second group scores
            pooled: Use pooled standard deviation (default: True)

        Returns:
            Cohen's d value
        """
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)

        if pooled:
            var1 = np.var(group1, ddof=1)
            var2 = np.var(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        else:
            pooled_std = np.std(np.concatenate([group1, group2]), ddof=1)

        d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        return float(d)

    @staticmethod
    def bootstrap_ci(
        data: np.ndarray,
        statistic=np.mean,
        ci: float = 0.95,
        n_bootstrap: int = 10000,
        seed: int = 42
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval.

        Args:
            data: Data array
            statistic: Function to compute (default: mean)
            ci: Confidence level (default 0.95 for 95%)
            n_bootstrap: Number of bootstrap samples
            seed: Random seed

        Returns:
            Tuple of (ci_lower, ci_upper)
        """
        rng = np.random.RandomState(seed)
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            sample = rng.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic(sample))

        bootstrap_stats = np.array(bootstrap_stats)

        # Percentile method
        alpha = 1 - ci
        ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

        return float(ci_lower), float(ci_upper)

    @staticmethod
    def bonferroni_correction(
        p_values: List[float],
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bonferroni multiple comparison correction.

        Adjust significance threshold: α' = α / m
        where m is number of comparisons.

        Args:
            p_values: List of p-values from multiple tests
            alpha: Original significance level

        Returns:
            (corrected_p_values, rejected_mask)
        """
        p_values = np.array(p_values)
        m = len(p_values)

        corrected_p = np.minimum(p_values * m, 1.0)
        rejected = corrected_p < alpha

        return corrected_p, rejected

    @staticmethod
    def benjamini_hochberg_correction(
        p_values: List[float],
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Benjamini-Hochberg False Discovery Rate control.

        Less conservative than Bonferroni, controls FDR instead of FWER.

        Args:
            p_values: List of p-values
            alpha: FDR control level

        Returns:
            (adjusted_p_values, rejected_mask)
        """
        p_values = np.array(p_values)
        m = len(p_values)

        # Sort p-values and keep track of original indices
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # Compute adjusted p-values
        adjusted_p = np.ones(m)
        for i, p in enumerate(sorted_p, 1):
            adjusted_p[sorted_idx[i-1]] = p * m / i

        # Ensure monotonicity
        for i in range(m - 2, -1, -1):
            adjusted_p[i] = min(adjusted_p[i], adjusted_p[i + 1])

        # Clip to [0, 1]
        adjusted_p = np.clip(adjusted_p, 0, 1)

        rejected = adjusted_p < alpha

        return adjusted_p, rejected


class ComprehensiveStatisticalSummary:
    """
    Generate comprehensive statistical summary report.
    """

    @staticmethod
    def evaluate_model(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Comprehensive statistical evaluation of predictions.

        Runs all tests and returns structured results.

        Args:
            y_true: True values
            y_pred: Predictions

        Returns:
            Dictionary with all results
        """
        tester = StatisticalTester()

        results = {
            'n_samples': len(y_true),
            'tests': {}
        }

        # Descriptive statistics
        results['y_true_mean'] = float(np.mean(y_true))
        results['y_true_std'] = float(np.std(y_true))
        results['y_pred_mean'] = float(np.mean(y_pred))
        results['y_pred_std'] = float(np.std(y_pred))
        results['prediction_error_mean'] = float(np.mean(y_pred - y_true))
        results['prediction_error_std'] = float(np.std(y_pred - y_true))

        # Statistical tests
        wilcoxon_result = tester.wilcoxon_signed_rank(y_true, y_pred)
        results['tests']['wilcoxon'] = {
            'p_value': wilcoxon_result.p_value,
            'significant_at_001': wilcoxon_result.significant_at_001,
            'effect_size': wilcoxon_result.effect_size,
        }

        t_test_result = tester.paired_t_test(y_true, y_pred)
        results['tests']['paired_t_test'] = {
            'p_value': t_test_result.p_value,
            'significant_at_001': t_test_result.significant_at_001,
            'effect_size': t_test_result.effect_size,  # Cohen's d
        }

        # Correlation
        if len(y_true) > 2:
            corr_result = tester.correlation_significance(y_true, y_pred, method='spearman')
            results['tests']['spearman_correlation'] = {
                'rho': corr_result.statistic,
                'p_value': corr_result.p_value,
                'significant_at_001': corr_result.significant_at_001,
            }

        return results


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)

    # Synthetic evaluation
    y_true = np.random.rand(100) * 0.5 + 0.25
    y_pred = y_true + np.random.randn(100) * 0.1

    tester = StatisticalTester()

    print("=== Wilcoxon Signed-Rank Test ===")
    wilcoxon_result = tester.wilcoxon_signed_rank(y_true, y_pred)
    print(wilcoxon_result)

    print("\n=== Paired t-test ===")
    t_result = tester.paired_t_test(y_true, y_pred)
    print(t_result)

    print("\n=== Spearman Correlation ===")
    corr_result = tester.correlation_significance(y_true, y_pred, method='spearman')
    print(corr_result)

    print("\n=== Cohen's d Effect Size ===")
    d = tester.cohens_d(y_true, y_pred)
    print(f"Cohen's d: {d:.4f}")

    print("\n=== Bootstrap CI for Mean ===")
    ci_lower, ci_upper = tester.bootstrap_ci(y_true - y_pred)
    print(f"95% CI on mean error: [{ci_lower:.6f}, {ci_upper:.6f}]")

    print("\n=== Comprehensive Summary ===")
    summary = ComprehensiveStatisticalSummary.evaluate_model(y_true, y_pred)
    print(f"N samples: {summary['n_samples']}")
    print(f"Wilcoxon p-value: {summary['tests']['wilcoxon']['p_value']:.6f}")
    print(f"Spearman rho: {summary['tests']['spearman_correlation']['rho']:.4f}")
