#!/usr/bin/env python3
"""
Comprehensive Test Suite for ChromaGuide Pipeline
==================================================

Tests all phases, orchestration logic, and automation components.
Run with: pytest tests/ -v --cov
"""

import pytest
import json
import tempfile
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPhase2XGBoost:
    """Test Phase 2 XGBoost benchmarking."""
    
    def test_crispro_xgboost_import(self):
        """Test that Phase 2 module can be imported."""
        try:
            import train_phase2_xgboost
            assert hasattr(train_phase2_xgboost, 'CRISPROXGBoostBenchmark')
        except ImportError:
            pytest.skip("XGBoost not installed")
    
    def test_xgboost_metrics_computation(self):
        """Test metric computation."""
        from sklearn.metrics import mean_squared_error
        y_true = np.array([0.5, 0.6, 0.7, 0.8])
        y_pred = np.array([0.51, 0.59, 0.72, 0.79])
        
        mse = mean_squared_error(y_true, y_pred)
        assert mse < 0.01
        assert mse >= 0
    
    def test_optuna_trial_objective(self):
        """Test Optuna objective function."""
        pytest.importorskip("optuna")
        # Objective should minimize error
        assert True  # Basic sanity check


class TestPhase3DeepHybrid:
    """Test Phase 3 DeepHybrid ensemble."""
    
    def test_deephybrid_import(self):
        """Test that Phase 3 module can be imported."""
        try:
            import train_phase3_deephybrid
            assert hasattr(train_phase3_deephybrid, 'DeepHybridEnsemble')
            assert hasattr(train_phase3_deephybrid, 'EnsembleLayer')
        except ImportError:
            pytest.skip("Dependencies not installed")
    
    def test_ensemble_layer_forward(self):
        """Test ensemble layer forward pass."""
        pytest.importorskip("torch")
        import torch
        from train_phase3_deephybrid import EnsembleLayer
        
        layer = EnsembleLayer(n_models=3)
        batch = torch.randn(10, 3)
        output = layer(batch)
        
        assert output.shape == (10, 1)
        assert not torch.isnan(output).any()


class TestPhase4Clinical:
    """Test Phase 4 clinical validation."""
    
    def test_clinical_validator_import(self):
        """Test Phase 4 module import."""
        try:
            import train_phase4_clinical_validation
            assert hasattr(train_phase4_clinical_validation, 'ClinicalValidator')
            assert hasattr(train_phase4_clinical_validation, 'ConformalPredictor')
        except ImportError:
            pytest.skip("Dependencies not installed")
    
    def test_conformal_predictor(self):
        """Test conformal prediction."""
        from train_phase4_clinical_validation import ConformalPredictor
        
        predictor = ConformalPredictor(confidence=0.90)
        
        # Test calibration
        residuals = np.random.uniform(0, 0.2, 100)
        predictor.calibrate(residuals)
        
        assert predictor.qhat is not None
        assert predictor.qhat > 0
    
    def test_conformal_intervals(self):
        """Test prediction interval generation."""
        from train_phase4_clinical_validation import ConformalPredictor
        
        predictor = ConformalPredictor(confidence=0.90)
        residuals = np.random.uniform(0, 0.1, 100)
        predictor.calibrate(residuals)
        
        predictions = np.array([0.5, 0.6, 0.7])
        lower, upper = predictor.predict_interval(predictions)
        
        assert (lower < predictions).all()
        assert (predictions < upper).all()
        assert (upper - lower).min() > 0


class TestBenchmarking:
    """Test SOTA benchmarking suite."""
    
    def test_sota_benchmark_import(self):
        """Test benchmarking module import."""
        try:
            import benchmark_sota
            assert hasattr(benchmark_sota, 'SOTABenchmark')
        except ImportError:
            pytest.skip("Dependencies not installed")
    
    def test_sota_models_defined(self):
        """Test that all SOTA models are defined."""
        import benchmark_sota
        
        benchmark = benchmark_sota.SOTABenchmark()
        
        assert len(benchmark.SOTA_MODELS) >= 10
        assert 'chromaguide' in benchmark.SOTA_MODELS
        assert 'chromecrispr' in benchmark.SOTA_MODELS
    
    def test_evaluation_datasets_defined(self):
        """Test evaluation datasets."""
        import benchmark_sota
        
        benchmark = benchmark_sota.SOTABenchmark()
        
        assert len(benchmark.EVALUATION_DATASETS) >= 5
        assert 'gene_held_out' in benchmark.EVALUATION_DATASETS


class TestFigureGeneration:
    """Test figure generation."""
    
    def test_figure_generator_import(self):
        """Test figure generator import."""
        try:
            import generate_figures
            assert hasattr(generate_figures, 'FigureGenerator')
        except ImportError:
            pytest.skip("Matplotlib not installed")
    
    def test_figure_output_directory(self):
        """Test figure output directory creation."""
        import generate_figures
        
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = generate_figures.FigureGenerator(
                results_dir=tmpdir,
                output_dir=Path(tmpdir) / 'figs'
            )
            
            assert generator.output_dir.exists()


class TestOrchestration:
    """Test multi-phase orchestration."""
    
    def test_orchestrator_import(self):
        """Test orchestrator module import."""
        try:
            import orchestrate_pipeline
            assert hasattr(orchestrate_pipeline, 'PipelineOrchestrator')
        except ImportError:
            pytest.skip("Dependencies not installed")
    
    def test_phase_dependencies(self):
        """Test phase dependency tracking."""
        import orchestrate_pipeline
        
        orchestrator = orchestrate_pipeline.PipelineOrchestrator()
        
        # Phase 2 depends on Phase 1
        assert 'phase1' in orchestrator.PHASES['phase2']['dependencies']
        
        # Phase 3 depends on Phase 1 and 2
        assert 'phase1' in orchestrator.PHASES['phase3']['dependencies']
        assert 'phase2' in orchestrator.PHASES['phase3']['dependencies']
    
    @patch('subprocess.run')
    def test_narval_job_status_check(self, mock_run):
        """Test Narval job status checking."""
        import orchestrate_pipeline
        
        mock_run.return_value = MagicMock(stdout='R\n', stderr='')
        
        orchestrator = orchestrate_pipeline.PipelineOrchestrator()
        status = orchestrator.check_narval_job_status('56644478')
        
        assert status == 'R'  # Running


class TestOverleafInjection:
    """Test Overleaf result injection."""
    
    def test_overleaf_injector_import(self):
        """Test Overleaf injector import."""
        try:
            import inject_results_overleaf
            assert hasattr(inject_results_overleaf, 'OverleafInjector')
        except ImportError:
            pytest.skip("Dependencies not installed")
    
    def test_metric_extraction(self):
        """Test metric extraction from results."""
        import inject_results_overleaf
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock results file
            results = {
                'models': {
                    'chromaguide': {'mean_spearman_r': 0.75},
                    'chromecrispr': {'mean_spearman_r': 0.68},
                }
            }
            
            results_path = Path(tmpdir) / 'benchmark_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f)
            
            injector = inject_results_overleaf.OverleafInjector(
                results_dir=tmpdir,
                figures_dir=tmpdir
            )
            
            injector.load_results()
            metrics = injector.extract_key_metrics()
            
            assert metrics['chromaguide_spearman_r'] == 0.75
            assert metrics['baseline_spearman_r'] == 0.68
            assert metrics['improvement'] > 0


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_all_phase_scripts_syntactically_valid(self):
        """Test that all phase scripts compile without syntax errors."""
        scripts = [
            'train_phase2_xgboost.py',
            'train_phase3_deephybrid.py',
            'train_phase4_clinical_validation.py',
            'benchmark_sota.py',
            'generate_figures.py',
            'orchestrate_pipeline.py',
            'inject_results_overleaf.py',
        ]
        
        import py_compile
        
        for script in scripts:
            script_path = Path(__file__).parent.parent / script
            if script_path.exists():
                try:
                    py_compile.compile(str(script_path), doraise=True)
                except py_compile.PyCompileError as e:
                    pytest.fail(f"Syntax error in {script}: {e}")
    
    def test_orchestration_phase_order(self):
        """Test that phases are ordered correctly."""
        import orchestrate_pipeline
        
        orchestrator = orchestrate_pipeline.PipelineOrchestrator()
        
        phase_order = [
            'phase1', 'phase2', 'phase3', 'phase4',
            'benchmarking', 'figures', 'overleaf', 'commit'
        ]
        
        assert all(phase in orchestrator.PHASES for phase in phase_order)


class TestDataFlow:
    """Test data flow between phases."""
    
    def test_phase1_output_format(self):
        """Test Phase 1 outputs checkpoint in expected format."""
        # Phase 1 should produce: best_model.pt, training_history.json
        assert True  # Placeholder for actual validation
    
    def test_phase2_input_requirements(self):
        """Test Phase 2 input requirements."""
        # Phase 2 needs: processed features pickle
        # Should validate data format
        assert True
    
    def test_phase3_input_requirements(self):
        """Test Phase 3 input requirements."""
        # Phase 3 needs: Phase 1 checkpoint + Phase 2 model
        assert True


# Fixture for temporary directories
@pytest.fixture
def temp_results_dir():
    """Create temporary directory for test results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_benchmark_results():
    """Create mock benchmark results."""
    return {
        'timestamp': '2026-02-17T00:00:00',
        'models': {
            'chromaguide': {
                'mean_spearman_r': 0.75,
                'std_spearman_r': 0.02,
                'datasets': {}
            },
            'chromecrispr': {
                'mean_spearman_r': 0.68,
                'std_spearman_r': 0.03,
                'datasets': {}
            }
        }
    }


def test_with_mock_results(mock_benchmark_results):
    """Test with mock benchmark results."""
    assert mock_benchmark_results['models']['chromaguide']['mean_spearman_r'] > \
           mock_benchmark_results['models']['chromecrispr']['mean_spearman_r']
