"""
MLflow experiment tracking and management.

Features:
- Experiment tracking and logging
- Parameter and metric logging
- Model artifact management
- Experiment comparison
- Hyperparameter search integration
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import logging


@dataclass
class ExperimentMetadata:
    """Metadata for experiment tracking."""
    experiment_name: str
    run_name: str
    phase: str
    model_type: str
    dataset: str
    timestamp: str
    git_commit: str = ""
    notes: str = ""


class MLflowTracker:
    """MLflow experiment tracking integration."""
    
    def __init__(self, experiment_name: str = "CRISPRO"):
        self.experiment_name = experiment_name
        self.run = None
        self.client = None
        self.experiment_id = None
        self.setup_client()
    
    def setup_client(self):
        """Initialize MLflow client."""
        try:
            import mlflow
            mlflow.set_experiment(self.experiment_name)
            self.client = mlflow.MlflowClient()
            self.experiment_id = self.client.get_experiment_by_name(
                self.experiment_name
            ).experiment_id
        except ImportError:
            logging.warning("MLflow not installed")
        except Exception as e:
            logging.error(f"MLflow setup failed: {e}")
    
    def start_run(self, run_name: str, tags: Optional[Dict] = None) -> str:
        """Start new MLflow run."""
        try:
            import mlflow
            self.run = mlflow.start_run(run_name=run_name)
            
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            return self.run.info.run_id
        except Exception as e:
            logging.error(f"Failed to start MLflow run: {e}")
            return None
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        try:
            import mlflow
            for key, value in params.items():
                mlflow.log_param(key, value)
        except Exception as e:
            logging.error(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics."""
        try:
            import mlflow
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logging.error(f"Failed to log metrics: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log artifact file."""
        try:
            import mlflow
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logging.error(f"Failed to log artifact: {e}")
    
    def log_model(self, model, artifact_path: str, framework: str = "pytorch"):
        """Log model artifact."""
        try:
            import mlflow
            if framework == "pytorch":
                mlflow.pytorch.log_model(model, artifact_path)
            elif framework == "sklearn":
                mlflow.sklearn.log_model(model, artifact_path)
            else:
                mlflow.log_model(model, artifact_path)
        except Exception as e:
            logging.error(f"Failed to log model: {e}")
    
    def end_run(self):
        """End current run."""
        try:
            import mlflow
            mlflow.end_run()
        except Exception as e:
            logging.error(f"Failed to end run: {e}")
    
    def get_run_id(self) -> str:
        """Get current run ID."""
        return self.run.info.run_id if self.run else None


class ExperimentComparison:
    """Compare multiple experiments."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.runs_data = []
    
    def fetch_experiments(self) -> List[Dict]:
        """Fetch all runs from experiment."""
        try:
            import mlflow
            client = mlflow.MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            
            runs = client.search_runs(experiment_id=experiment.experiment_id)
            self.runs_data = [
                {
                    'run_id': run.info.run_id,
                    'status': run.info.status,
                    'params': run.data.params,
                    'metrics': run.data.metrics
                }
                for run in runs
            ]
            return self.runs_data
        except Exception as e:
            logging.error(f"Failed to fetch experiments: {e}")
            return []
    
    def compare_metrics(self, metric_names: List[str]) -> Dict:
        """Compare specific metrics across runs."""
        comparison = {metric: [] for metric in metric_names}
        
        for run in self.runs_data:
            for metric in metric_names:
                if metric in run['metrics']:
                    comparison[metric].append({
                        'run_id': run['run_id'],
                        'value': run['metrics'][metric]
                    })
        
        return comparison
    
    def get_best_run(self, metric: str, mode: str = 'max') -> Dict:
        """Get best run for metric."""
        best_run = None
        best_value = float('-inf') if mode == 'max' else float('inf')
        
        for run in self.runs_data:
            if metric in run['metrics']:
                value = run['metrics'][metric]
                if (mode == 'max' and value > best_value) or \
                   (mode == 'min' and value < best_value):
                    best_value = value
                    best_run = run
        
        return best_run


class ExperimentLogger:
    """Structured experiment logging."""
    
    def __init__(self, log_dir: Path = Path("logs/experiments")):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def save_experiment_config(self, config: Dict, exp_name: str):
        """Save experiment configuration."""
        config_path = self.log_dir / f"{exp_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return config_path
    
    def load_experiment_config(self, exp_name: str) -> Dict:
        """Load experiment configuration."""
        config_path = self.log_dir / f"{exp_name}_config.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def save_results(self, results: Dict, exp_name: str):
        """Save experiment results."""
        results_path = self.log_dir / f"{exp_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        return results_path
    
    def save_experiment_history(self, history: List[Dict], exp_name: str):
        """Save training history."""
        history_path = self.log_dir / f"{exp_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        return history_path


class HyperparameterStudy:
    """Track hyperparameter search studies."""
    
    def __init__(self, study_name: str):
        self.study_name = study_name
        self.trials = []
        self.best_trial = None
    
    def add_trial(self, trial_id: int, params: Dict, metrics: Dict):
        """Record trial."""
        self.trials.append({
            'trial_id': trial_id,
            'params': params,
            'metrics': metrics,
            'timestamp': str(Path.cwd())
        })
    
    def get_best_trial(self, metric: str, mode: str = 'max') -> Dict:
        """Get best trial."""
        best = None
        best_value = float('-inf') if mode == 'max' else float('inf')
        
        for trial in self.trials:
            if metric in trial['metrics']:
                value = trial['metrics'][metric]
                if (mode == 'max' and value > best_value) or \
                   (mode == 'min' and value < best_value):
                    best_value = value
                    best = trial
        
        self.best_trial = best
        return best
    
    def get_trial_statistics(self, metric: str) -> Dict:
        """Calculate statistics for metric."""
        values = [t['metrics'].get(metric) for t in self.trials 
                 if metric in t['metrics']]
        
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
