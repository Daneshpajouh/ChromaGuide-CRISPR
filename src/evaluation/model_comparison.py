"""
Model comparison and automated reporting.

Features:
- Compare multiple model runs
- Generate comparison reports
- Statistical significance testing
- Visualization of results
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy import stats
import json


@dataclass
class ModelResult:
    """Single model result."""
    model_name: str
    metrics: Dict[str, float]
    params: Dict[str, any]
    runtime: float
    timestamp: str


class ModelComparison:
    """Compare multiple model results."""
    
    def __init__(self):
        self.results: List[ModelResult] = []
        self.comparison_df = None
    
    def add_result(self, result: ModelResult):
        """Add model result."""
        self.results.append(result)
    
    def add_results_from_dict(self, results_dict: Dict):
        """Load results from dictionary."""
        for model_name, model_data in results_dict.items():
            result = ModelResult(
                model_name=model_name,
                metrics=model_data.get('metrics', {}),
                params=model_data.get('params', {}),
                runtime=model_data.get('runtime', 0),
                timestamp=model_data.get('timestamp', '')
            )
            self.add_result(result)
    
    def create_comparison_df(self) -> pd.DataFrame:
        """Create comparison dataframe."""
        data = []
        for result in self.results:
            row = {'model': result.model_name, 'runtime': result.runtime}
            row.update(result.metrics)
            data.append(row)
        
        self.comparison_df = pd.DataFrame(data)
        return self.comparison_df
    
    def get_best_model(self, metric: str, mode: str = 'max') -> str:
        """Get best model for metric."""
        if self.comparison_df is None:
            self.create_comparison_df()
        
        if mode == 'max':
            best_idx = self.comparison_df[metric].idxmax()
        else:
            best_idx = self.comparison_df[metric].idxmin()
        
        return self.comparison_df.loc[best_idx, 'model']
    
    def get_ranking(self, metric: str, mode: str = 'max') -> List[Tuple[str, float]]:
        """Rank models by metric."""
        if self.comparison_df is None:
            self.create_comparison_df()
        
        if mode == 'max':
            ranking = self.comparison_df.nlargest(len(self.results), metric)
        else:
            ranking = self.comparison_df.nsmallest(len(self.results), metric)
        
        return list(zip(ranking['model'], ranking[metric]))
    
    def statistical_significance(self, metric: str, alpha: float = 0.05) -> Dict:
        """Test statistical significance with ANOVA."""
        if len(self.results) < 2:
            return {}
        
        metric_values = [r.metrics.get(metric, 0) for r in self.results]
        
        if len(set(metric_values)) == 1:
            return {'significant': False, 'p_value': 1.0}
        
        f_stat, p_value = stats.f_oneway(*[[v] for v in metric_values])
        
        return {
            'significant': p_value < alpha,
            'p_value': p_value,
            'f_statistic': f_stat
        }
    
    def pairwise_comparison(self, metric: str) -> Dict:
        """Pairwise comparison between models."""
        comparisons = {}
        model_names = [r.model_name for r in self.results]
        metric_values = [r.metrics.get(metric, 0) for r in self.results]
        
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if i < j:
                    key = f"{m1} vs {m2}"
                    diff = metric_values[i] - metric_values[j]
                    comparisons[key] = diff
        
        return comparisons


class ComparisonReport:
    """Generate comparison report."""
    
    def __init__(self, comparison: ModelComparison):
        self.comparison = comparison
        self.report = {}
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics."""
        if self.comparison.comparison_df is None:
            self.comparison.create_comparison_df()
        
        df = self.comparison.comparison_df
        
        summary = {
            'num_models': len(self.comparison.results),
            'metrics_summary': {}
        }
        
        for col in df.columns:
            if col != 'model':
                summary['metrics_summary'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        self.report['summary'] = summary
        return summary
    
    def generate_rankings(self, metrics: List[str]) -> Dict:
        """Generate rankings for all metrics."""
        rankings = {}
        for metric in metrics:
            rankings[metric] = {
                'ascending': self.comparison.get_ranking(metric, mode='min'),
                'descending': self.comparison.get_ranking(metric, mode='max')
            }
        self.report['rankings'] = rankings
        return rankings
    
    def generate_statistical_tests(self, metrics: List[str]) -> Dict:
        """Generate statistical test results."""
        tests = {}
        for metric in metrics:
            tests[metric] = self.comparison.statistical_significance(metric)
        self.report['statistical_tests'] = tests
        return tests
    
    def to_json(self, filepath: Path):
        """Export report to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)
    
    def to_markdown(self, filepath: Path):
        """Export report to Markdown."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write("# Model Comparison Report\n\n")
            
            if 'summary' in self.report:
                f.write("## Summary Statistics\n\n")
                f.write(f"- Number of models: {self.report['summary']['num_models']}\n\n")
                
                f.write("### Metrics\n\n")
                for metric, stats_dict in self.report['summary']['metrics_summary'].items():
                    f.write(f"#### {metric}\n\n")
                    f.write(f"- Mean: {stats_dict['mean']:.4f}\n")
                    f.write(f"- Std: {stats_dict['std']:.4f}\n")
                    f.write(f"- Min: {stats_dict['min']:.4f}\n")
                    f.write(f"- Max: {stats_dict['max']:.4f}\n\n")


class PerformanceBenchmark:
    """Benchmark model performance."""
    
    def __init__(self, test_data_size: int = 1000):
        self.test_data_size = test_data_size
        self.benchmarks = {}
    
    def benchmark_model(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        model_name: str
    ) -> Dict:
        """Benchmark single model."""
        import time
        
        # Inference time
        start = time.time()
        for _ in range(10):
            _ = model.predict(X_test[:100])
        inference_time = (time.time() - start) / 10
        
        # Accuracy
        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        
        # Memory footprint
        import sys
        model_size = sys.getsizeof(model) / 1024 / 1024  # MB
        
        benchmark = {
            'model_name': model_name,
            'inference_time_ms': inference_time * 1000,
            'accuracy': accuracy,
            'model_size_mb': model_size
        }
        
        self.benchmarks[model_name] = benchmark
        return benchmark
    
    def compare_benchmarks(self) -> pd.DataFrame:
        """Compare all benchmarks."""
        return pd.DataFrame(list(self.benchmarks.values()))
