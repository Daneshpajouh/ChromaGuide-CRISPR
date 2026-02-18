"""
Hyperparameter optimization using Optuna and Ray Tune.

Implements:
- Optuna-based Bayesian optimization
- Ray Tune distributed hyperparameter search
- Population-based training (PBT)
- Hyperband scheduling
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional, Tuple, List
import json
import warnings


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter optimization."""
    optimizer: str = 'optuna'  # optuna or ray_tune
    n_trials: int = 100
    timeout: int = 3600
    n_jobs: int = 1
    sampler: str = 'tpe'  # Tree-structured Parzen Estimator
    pruner: str = 'median'
    seed: int = 42
    verbose: int = 1


class OptunaOptimizer:
    """Optuna-based hyperparameter optimization."""
    
    def __init__(self, config: HyperparameterConfig):
        self.config = config
        self.study = None
        self.best_params = None
        self.best_value = None
        self.trials_history = []
    
    def create_study(self, direction: str = 'minimize'):
        """Create Optuna study."""
        try:
            import optuna
            from optuna.samplers import TPESampler
            from optuna.pruners import MedianPruner
            
            sampler = TPESampler(seed=self.config.seed)
            pruner = MedianPruner()
            
            self.study = optuna.create_study(
                direction=direction,
                sampler=sampler,
                pruner=pruner
            )
        except ImportError:
            raise ImportError("Optuna not installed. Install with: pip install optuna")
    
    def optimize(
        self,
        objective_fn: Callable,
        n_trials: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run optimization."""
        if self.study is None:
            self.create_study()
        
        n_trials = n_trials or self.config.n_trials
        
        self.study.optimize(
            objective_fn,
            n_trials=n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=self.config.verbose > 0
        )
        
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        self.trials_history = self.study.trials
        
        return self.best_params
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """Get optimization history."""
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(self.trials_history),
            'trial_values': [t.value for t in self.trials_history if t.value is not None]
        }
    
    def suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """Example hyperparameter suggestions."""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'num_layers': trial.suggest_int('num_layers', 2, 8),
            'hidden_dim': trial.suggest_int('hidden_dim', 64, 512, step=64),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        }


class RayTuneOptimizer:
    """Ray Tune distributed hyperparameter optimization."""
    
    def __init__(self, config: HyperparameterConfig):
        self.config = config
        self.best_config = None
        self.best_value = None
        self.analysis = None
    
    def optimize(
        self,
        training_fn: Callable,
        search_space: Dict[str, Any],
        metric: str = 'validation_loss',
        mode: str = 'min',
    ) -> Dict[str, Any]:
        """Run distributed optimization with Ray Tune."""
        try:
            from ray import tune
            from ray.tune.schedulers import HyperBandScheduler
            
            scheduler = HyperBandScheduler(
                time_attr='training_iteration',
                metric=metric,
                mode=mode,
            )
            
            self.analysis = tune.run(
                training_fn,
                name='hyperparameter_search',
                config=search_space,
                num_samples=self.config.n_trials,
                scheduler=scheduler,
                verbose=self.config.verbose,
                progress_reporter=tune.CLIReporter(),
            )
            
            self.best_config = self.analysis.best_config
            self.best_value = self.analysis.best_result[metric]
            
            return self.best_config
            
        except ImportError:
            raise ImportError("Ray Tune not installed. Install with: pip install ray[tune]")
    
    def get_best_trial(self) -> Dict[str, Any]:
        """Get best trial results."""
        if self.analysis is None:
            return None
        
        return {
            'config': self.best_config,
            'metric': self.best_value,
            'logdir': self.analysis.best_logdir
        }


class PopulationBasedTraining:
    """Population-based training for hyperparameter optimization."""
    
    def __init__(self, population_size: int = 5, generations: int = 10):
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.fitnesses = []
    
    def initialize_population(self, hp_bounds: Dict[str, Tuple]) -> List[Dict]:
        """Initialize random population."""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for hp_name, (low, high) in hp_bounds.items():
                individual[hp_name] = np.random.uniform(low, high)
            population.append(individual)
        self.population = population
        return population
    
    def evaluate_fitness(self, evaluation_fn: Callable) -> List[float]:
        """Evaluate population fitness."""
        self.fitnesses = [evaluation_fn(ind) for ind in self.population]
        return self.fitnesses
    
    def select_top(self, k: int = 2) -> List[Dict]:
        """Select top k individuals."""
        top_indices = np.argsort(self.fitnesses)[-k:]
        return [self.population[i] for i in top_indices]
    
    def mutate(self, individual: Dict, mutation_rate: float = 0.1) -> Dict:
        """Mutate individual hyperparameters."""
        mutated = individual.copy()
        for key, value in mutated.items():
            if np.random.random() < mutation_rate:
                noise = np.random.normal(0, 0.1 * value)
                mutated[key] = max(0, value + noise)
        return mutated
    
    def run_pbt(self, evaluation_fn: Callable, hp_bounds: Dict) -> Dict:
        """Run population-based training."""
        self.initialize_population(hp_bounds)
        
        for generation in range(self.generations):
            self.evaluate_fitness(evaluation_fn)
            
            # Select top performers
            top = self.select_top(k=max(1, self.population_size // 3))
            
            # Mutate and fill population
            new_population = top.copy()
            while len(new_population) < self.population_size:
                parent = np.random.choice(top)
                child = self.mutate(parent)
                new_population.append(child)
            
            self.population = new_population[:self.population_size]
        
        # Final evaluation
        best_idx = np.argmax(self.evaluate_fitness(evaluation_fn))
        return self.population[best_idx]


class HyperbandScheduler:
    """Hyperband multi-fidelity optimization."""
    
    def __init__(self, max_iter: int = 81, eta: int = 3):
        self.max_iter = max_iter
        self.eta = eta
        self.brackets = []
    
    def get_brackets(self) -> List[Dict]:
        """Calculate Hyperband brackets."""
        s_max = int(np.log(self.max_iter) / np.log(self.eta))
        B = (s_max + 1) * self.max_iter
        
        brackets = []
        for s in range(s_max, -1, -1):
            n = int(np.ceil(B / self.max_iter / (s + 1) * self.eta ** s))
            r = self.max_iter * self.eta ** (-s)
            
            brackets.append({
                'stage': s,
                'num_configs': n,
                'resources': r,
            })
        
        return brackets
    
    def run_bracket(self, configs: List, evaluation_fn: Callable) -> Tuple[Dict, float]:
        """Run single bracket."""
        best_config = None
        best_score = float('inf')
        
        for config in configs:
            score = evaluation_fn(config)
            if score < best_score:
                best_score = score
                best_config = config
        
        return best_config, best_score


class HyperparameterScheduler:
    """Learning rate and momentum scheduling."""
    
    def __init__(self, strategy: str = 'cosine'):
        self.strategy = strategy
        self.schedule = []
    
    def cosine_annealing(
        self, 
        initial_lr: float, 
        min_lr: float, 
        epochs: int
    ) -> List[float]:
        """Cosine annealing schedule."""
        schedule = []
        for epoch in range(epochs):
            lr = min_lr + (initial_lr - min_lr) * (
                1 + np.cos(np.pi * epoch / epochs)
            ) / 2
            schedule.append(lr)
        return schedule
    
    def exponential_decay(
        self,
        initial_lr: float,
        decay_rate: float,
        epochs: int
    ) -> List[float]:
        """Exponential decay schedule."""
        return [initial_lr * (decay_rate ** (e / epochs)) for e in range(epochs)]
    
    def step_decay(
        self,
        initial_lr: float,
        decay_rate: float,
        step_size: int,
        epochs: int
    ) -> List[float]:
        """Step decay schedule."""
        schedule = []
        for epoch in range(epochs):
            lr = initial_lr * (decay_rate ** (epoch // step_size))
            schedule.append(lr)
        return schedule
    
    def warm_up_then_decay(
        self,
        initial_lr: float,
        warmup_epochs: int,
        total_epochs: int
    ) -> List[float]:
        """Warmup then decay schedule."""
        schedule = []
        for epoch in range(total_epochs):
            if epoch < warmup_epochs:
                lr = initial_lr * (epoch / warmup_epochs)
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                lr = initial_lr * (1 + np.cos(np.pi * progress)) / 2
            schedule.append(lr)
        return schedule


class GridSearchOptimizer:
    """Simple grid search optimizer."""
    
    def __init__(self, param_grid: Dict[str, List]):
        self.param_grid = param_grid
        self.results = []
    
    def generate_combinations(self) -> List[Dict]:
        """Generate all parameter combinations."""
        import itertools
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        return combinations
    
    def optimize(self, evaluation_fn: Callable) -> Dict:
        """Run grid search."""
        combinations = self.generate_combinations()
        
        best_params = None
        best_value = float('inf')
        
        for params in combinations:
            value = evaluation_fn(params)
            self.results.append({'params': params, 'value': value})
            
            if value < best_value:
                best_value = value
                best_params = params
        
        return best_params


def create_optimizer(config: HyperparameterConfig) -> OptunaOptimizer:
    """Factory function for creating optimizer."""
    if config.optimizer == 'optuna':
        return OptunaOptimizer(config)
    elif config.optimizer == 'ray_tune':
        return RayTuneOptimizer(config)
    else:
        return OptunaOptimizer(config)
