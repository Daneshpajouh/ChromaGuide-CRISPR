"""
Comprehensive benchmarking suite against SOTA models.

Features:
- Benchmark runner
- SOTA comparison
- Performance metrics
- Automated report generation
"""

import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple
import json


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    model_name: str
    dataset: str
    metric_name: str
    metric_value: float
    inference_time_ms: float
    memory_mb: float
    timestamp: str


class BenchmarkSuite:
    """Comprehensive benchmarking suite."""
    
    def __init__(self, results_dir: Path = Path("benchmarks")):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    def benchmark_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        dataset_name: str,
        metric_fn: Optional[Callable] = None
    ) -> BenchmarkResult:
        """Benchmark single model."""
        import sys
        from datetime import datetime
        
        # Inference time
        start = time.time()
        predictions = model.predict(X_test)
        inference_time = (time.time() - start) * 1000  # ms
        
        # Memory
        model_size = sys.getsizeof(model) / 1024 / 1024
        
        # Metric
        if metric_fn:
            metric_value = metric_fn(y_test, predictions)
        else:
            metric_value = np.mean(predictions == y_test)
        
        result = BenchmarkResult(
            model_name=model_name,
            dataset=dataset_name,
            metric_name="accuracy",
            metric_value=metric_value,
            inference_time_ms=inference_time,
            memory_mb=model_size,
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        return result
    
    def compare_models(
        self,
        models: Dict[str, any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        metric_fn: Optional[Callable] = None
    ) -> Dict:
        """Compare multiple models on same dataset."""
        comparison = {}
        
        for model_name, model in models.items():
            result = self.benchmark_model(
                model, X_test, y_test,
                model_name, "benchmark_dataset",
                metric_fn
            )
            comparison[model_name] = {
                'accuracy': result.metric_value,
                'inference_time_ms': result.inference_time_ms,
                'memory_mb': result.memory_mb
            }
        
        return comparison
    
    def scale_to_dataset_size(self, X_test: np.ndarray) -> Dict:
        """Benchmark scaling with different dataset sizes."""
        scaling_results = {}
        
        for size_ratio in [0.1, 0.5, 1.0]:
            size = int(len(X_test) * size_ratio)
            subset = X_test[:size]
            
            start = time.time()
            _ = self.model.predict(subset)
            elapsed = (time.time() - start) * 1000
            
            scaling_results[f"{size_ratio*100:.0f}%"] = elapsed
        
        return scaling_results
    
    def export_results(self, filepath: Path):
        """Export benchmark results."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        results_dict = [
            {
                'model': r.model_name,
                'dataset': r.dataset,
                'metric': r.metric_value,
                'inference_time_ms': r.inference_time_ms,
                'memory_mb': r.memory_mb,
                'timestamp': r.timestamp
            }
            for r in self.results
        ]
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)


class SOTAComparison:
    """Compare against SOTA models."""
    
    SOTA_MODELS = {
        'DeepHybrid': {'accuracy': 0.92, 'f1': 0.89},
        'TransCRISPR': {'accuracy': 0.88, 'f1': 0.85},
        'CRISPRon': {'accuracy': 0.86, 'f1': 0.83},
        'XGBoost': {'accuracy': 0.84, 'f1': 0.81},
    }
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, model_name: str, metrics: Dict[str, float]):
        """Add model result."""
        self.results[model_name] = metrics
    
    def get_comparison(self) -> Dict:
        """Get comparison with SOTA."""
        comparison = {
            'sota_benchmarks': self.SOTA_MODELS,
            'our_results': self.results,
            'improvements': {}
        }
        
        for metric in ['accuracy', 'f1']:
            sota_values = [m.get(metric, 0) for m in self.SOTA_MODELS.values()]
            our_values = [m.get(metric, 0) for m in self.results.values()]
            
            if sota_values and our_values:
                sota_mean = np.mean(sota_values)
                our_mean = np.mean(our_values)
                improvement = ((our_mean - sota_mean) / sota_mean) * 100
                
                comparison['improvements'][metric] = {
                    'absolute': our_mean - sota_mean,
                    'percent': improvement
                }
        
        return comparison
    
    def generate_comparison_table(self) -> str:
        """Generate markdown comparison table."""
        comparison = self.get_comparison()
        
        table = "| Model | Accuracy | F1-Score |\n"
        table += "|-------|----------|----------|\n"
        
        for model, metrics in self.SOTA_MODELS.items():
            table += f"| {model} | {metrics.get('accuracy', 0):.4f} | {metrics.get('f1', 0):.4f} |\n"
        
        table += "\n## Our Results\n\n"
        table += "| Model | Accuracy | F1-Score |\n"
        table += "|-------|----------|----------|\n"
        
        for model, metrics in self.results.items():
            table += f"| {model} | {metrics.get('accuracy', 0):.4f} | {metrics.get('f1', 0):.4f} |\n"
        
        return table


class PerformanceProfiler:
    """Profile model performance characteristics."""
    
    def __init__(self):
        self.profiles = {}
    
    def profile_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> Dict:
        """Profile model performance."""
        profile = {
            'name': model_name,
            'inference_stats': {},
            'memory_stats': {},
            'accuracy_stats': {}
        }
        
        # Inference timing
        times = []
        for _ in range(10):
            start = time.time()
            predictions = model.predict(X_test)
            times.append((time.time() - start) * 1000)
        
        profile['inference_stats'] = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times)
        }
        
        # Accuracy
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        
        profile['accuracy_stats'] = {
            'accuracy': acc,
            'f1_score': f1
        }
        
        self.profiles[model_name] = profile
        return profile
    
    def compare_profiles(self) -> Dict:
        """Compare all profiles."""
        return {
            'profiles': self.profiles,
            'best_accuracy': max(
                self.profiles.values(),
                key=lambda x: x['accuracy_stats'].get('accuracy', 0)
            )['name']
        }


class BenchmarkReport:
    """Generate benchmarking report."""
    
    def __init__(self):
        self.report = {}
    
    def add_section(self, section_name: str, content: Dict):
        """Add report section."""
        self.report[section_name] = content
    
    def generate_markdown(self) -> str:
        """Generate markdown report."""
        md = "# Benchmarking Report\n\n"
        
        for section, content in self.report.items():
            md += f"## {section}\n\n"
            
            if isinstance(content, dict):
                for key, value in content.items():
                    md += f"- **{key}**: {value}\n"
            elif isinstance(content, str):
                md += content
            
            md += "\n"
        
        return md
    
    def save_report(self, filepath: Path):
        """Save report to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        md = self.generate_markdown()
        with open(filepath, 'w') as f:
            f.write(md)
