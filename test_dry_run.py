#!/usr/bin/env python3
"""Local dry-run test for ChromaGuide v2.

Tests each backbone on CPU with tiny synthetic data to verify:
1. All imports work
2. Model builds correctly
3. Forward pass succeeds
4. Loss computes
5. Backward pass succeeds
6. Conformal predictor works
7. All evaluation metrics compute

Usage:
    python test_dry_run.py
"""
import sys
import os
import traceback
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

BACKBONES = ['cnn_gru', 'dnabert2', 'caduceus', 'evo', 'nucleotide_transformer']
PASS = "✓ PASS"
FAIL = "✗ FAIL"


def create_synthetic_batch(batch_size=4, seq_len=23, n_tracks=3, n_bins=100):
    """Create a synthetic batch mimicking CRISPRDataset output."""
    bases = 'ACGT'
    sequences_str = []
    one_hot = torch.zeros(batch_size, 4, seq_len)
    
    for b in range(batch_size):
        seq = ''.join(np.random.choice(list(bases)) for _ in range(seq_len))
        sequences_str.append(seq)
        for i, base in enumerate(seq):
            one_hot[b, 'ACGT'.index(base), i] = 1.0
    
    return {
        'sequence': one_hot,
        'sequence_str': sequences_str,
        'epigenomic': torch.randn(batch_size, n_tracks, n_bins),
        'efficacy': torch.rand(batch_size),
    }


def build_config_for_backbone(backbone):
    """Build a minimal OmegaConf config for testing."""
    from omegaconf import OmegaConf
    
    # Load default config
    default_path = PROJECT_ROOT / 'chromaguide' / 'configs' / 'default.yaml'
    cfg = OmegaConf.load(default_path)
    
    # Override backbone type
    cfg.model.sequence_encoder.type = backbone
    
    # Load backbone-specific config if exists
    backbone_config = PROJECT_ROOT / 'chromaguide' / 'configs' / f'{backbone}.yaml'
    if backbone_config.exists():
        exp_cfg = OmegaConf.load(backbone_config)
        cfg = OmegaConf.merge(cfg, exp_cfg)
    
    # Force CPU and small settings
    cfg.project.device = 'cpu'
    cfg.project.precision = 32  # No AMP on CPU
    cfg.project.num_workers = 0
    cfg.training.batch_size = 4
    
    OmegaConf.resolve(cfg)
    return cfg


def test_imports():
    """Test all critical imports."""
    results = {}
    
    modules_to_import = [
        ('chromaguide', None),
        ('chromaguide.modules', None),
        ('chromaguide.modules.sequence_encoders', 'build_sequence_encoder'),
        ('chromaguide.modules.epigenomic_encoder', 'EpigenomicEncoder'),
        ('chromaguide.modules.fusion', 'build_fusion'),
        ('chromaguide.modules.prediction_head', 'BetaRegressionHead'),
        ('chromaguide.modules.conformal', 'SplitConformalPredictor'),
        ('chromaguide.models.chromaguide', 'ChromaGuideModel'),
        ('chromaguide.training.losses', 'CalibratedLoss'),
        ('chromaguide.training.trainer', 'CosineWarmupScheduler'),
        ('chromaguide.utils.reproducibility', 'set_seed'),
        ('chromaguide.data.dataset', 'CRISPRDataset'),
        ('chromaguide.data.splits', 'SplitBuilder'),
    ]
    
    for module_name, attr_name in modules_to_import:
        try:
            mod = __import__(module_name, fromlist=[attr_name] if attr_name else [])
            if attr_name:
                getattr(mod, attr_name)
            results[f"import {module_name}{'.' + attr_name if attr_name else ''}"] = PASS
        except Exception as e:
            results[f"import {module_name}{'.' + attr_name if attr_name else ''}"] = f"{FAIL}: {e}"
    
    return results


def test_backbone(backbone, verbose=True):
    """Test a single backbone end-to-end on CPU."""
    results = {}
    prefix = f"[{backbone}]"
    
    try:
        cfg = build_config_for_backbone(backbone)
        results[f"{prefix} config"] = PASS
    except Exception as e:
        results[f"{prefix} config"] = f"{FAIL}: {e}"
        return results
    
    # Build model
    try:
        from chromaguide.models.chromaguide import ChromaGuideModel
        model = ChromaGuideModel(cfg)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results[f"{prefix} build model ({trainable_params:,} params)"] = PASS
    except Exception as e:
        results[f"{prefix} build model"] = f"{FAIL}: {e}"
        traceback.print_exc()
        return results
    
    # Forward pass
    try:
        batch = create_synthetic_batch()
        model.eval()
        
        needs_raw = backbone in ['dnabert2', 'evo', 'nucleotide_transformer']
        raw_seqs = batch['sequence_str'] if needs_raw else None
        
        with torch.no_grad():
            mu, phi = model(batch['sequence'], batch['epigenomic'], raw_sequences=raw_seqs)
        
        assert mu.shape == (4, 1), f"mu shape: {mu.shape}"
        assert phi.shape == (4, 1), f"phi shape: {phi.shape}"
        assert (mu > 0).all() and (mu < 1).all(), f"mu out of range: {mu}"
        assert (phi > 0).all(), f"phi not positive: {phi}"
        
        results[f"{prefix} forward pass (mu={mu.mean():.4f}, phi={phi.mean():.2f})"] = PASS
    except Exception as e:
        results[f"{prefix} forward pass"] = f"{FAIL}: {e}"
        traceback.print_exc()
        return results
    
    # Loss computation
    try:
        from chromaguide.training.losses import CalibratedLoss
        criterion = CalibratedLoss(
            primary_type=cfg.training.loss.primary,
            lambda_cal=cfg.training.loss.lambda_cal,
        )
        
        model.train()
        mu, phi = model(batch['sequence'], batch['epigenomic'], raw_sequences=raw_seqs)
        losses = criterion(mu, phi, batch['efficacy'])
        
        assert 'total' in losses
        assert losses['total'].requires_grad
        assert not torch.isnan(losses['total']), "Loss is NaN!"
        assert not torch.isinf(losses['total']), "Loss is Inf!"
        
        results[f"{prefix} loss (total={losses['total'].item():.4f})"] = PASS
    except Exception as e:
        results[f"{prefix} loss computation"] = f"{FAIL}: {e}"
        traceback.print_exc()
        return results
    
    # Backward pass
    try:
        losses['total'].backward()
        
        # Check gradients exist
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
        assert has_grad, "No gradients computed!"
        
        # Check for NaN gradients
        nan_grads = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
        assert nan_grads == 0, f"{nan_grads} parameters have NaN gradients!"
        
        results[f"{prefix} backward pass"] = PASS
    except Exception as e:
        results[f"{prefix} backward pass"] = f"{FAIL}: {e}"
        traceback.print_exc()
    
    # Optimizer step
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.step()
        optimizer.zero_grad()
        results[f"{prefix} optimizer step"] = PASS
    except Exception as e:
        results[f"{prefix} optimizer step"] = f"{FAIL}: {e}"
    
    # Conformal prediction
    try:
        from chromaguide.modules.conformal import SplitConformalPredictor
        
        # Generate some fake predictions
        n_cal, n_test = 100, 50
        cal_mu = np.random.rand(n_cal).astype(np.float32) * 0.8 + 0.1
        cal_phi = np.random.rand(n_cal).astype(np.float32) * 50 + 5
        cal_y = cal_mu + np.random.randn(n_cal).astype(np.float32) * 0.05
        cal_y = np.clip(cal_y, 0.01, 0.99)
        
        test_mu = np.random.rand(n_test).astype(np.float32) * 0.8 + 0.1
        test_phi = np.random.rand(n_test).astype(np.float32) * 50 + 5
        test_y = test_mu + np.random.randn(n_test).astype(np.float32) * 0.05
        test_y = np.clip(test_y, 0.01, 0.99)
        
        cp = SplitConformalPredictor(alpha=0.10, use_beta_sigma=True, tolerance=0.02)
        q_hat = cp.calibrate(cal_y, cal_mu, cal_phi)
        lower, upper = cp.predict(test_mu, test_phi)
        stats = cp.evaluate_coverage(test_y, lower, upper)
        
        results[f"{prefix} conformal (coverage={stats['coverage']:.2f}, width={stats['avg_width']:.4f})"] = PASS
    except Exception as e:
        results[f"{prefix} conformal prediction"] = f"{FAIL}: {e}"
        traceback.print_exc()
    
    # Scheduler test
    try:
        from chromaguide.training.trainer import CosineWarmupScheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=5, max_epochs=100, min_lr=1e-6)
        for _ in range(10):
            scheduler.step()
        results[f"{prefix} scheduler"] = PASS
    except Exception as e:
        results[f"{prefix} scheduler"] = f"{FAIL}: {e}"
    
    return results


def test_data_collation():
    """Test that data batching works correctly with string fields."""
    results = {}
    
    try:
        from torch.utils.data import DataLoader, Dataset
        
        class DummyDataset(Dataset):
            def __init__(self, n=20):
                self.n = n
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                seq = ''.join(np.random.choice(list('ACGT')) for _ in range(23))
                one_hot = torch.zeros(4, 23)
                for i, b in enumerate(seq):
                    one_hot['ACGT'.index(b), i] = 1.0
                return {
                    'sequence': one_hot,
                    'sequence_str': seq,
                    'epigenomic': torch.randn(3, 100),
                    'efficacy': torch.tensor(np.random.rand(), dtype=torch.float32),
                    'cell_line': 'HEK293T',
                    'gene': 'TP53',
                }
        
        ds = DummyDataset(20)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        
        batch = next(iter(loader))
        
        assert batch['sequence'].shape == (4, 4, 23)
        assert isinstance(batch['sequence_str'], list) and len(batch['sequence_str']) == 4
        assert batch['epigenomic'].shape == (4, 3, 100)
        assert batch['efficacy'].shape == (4,)
        assert isinstance(batch['cell_line'], list) and len(batch['cell_line']) == 4
        
        results["data collation with strings"] = PASS
    except Exception as e:
        results["data collation with strings"] = f"{FAIL}: {e}"
        traceback.print_exc()
    
    return results


def main():
    print("=" * 70)
    print("ChromaGuide v2 — Local Dry-Run Test Suite")
    print("=" * 70)
    
    all_results = {}
    
    # 1. Test imports
    print("\n--- Testing Imports ---")
    import_results = test_imports()
    all_results.update(import_results)
    for k, v in import_results.items():
        status = "PASS" if "PASS" in v else "FAIL"
        print(f"  {v}  {k}")
    
    # 2. Test data collation
    print("\n--- Testing Data Collation ---")
    collation_results = test_data_collation()
    all_results.update(collation_results)
    for k, v in collation_results.items():
        print(f"  {v}  {k}")
    
    # 3. Test each backbone
    for backbone in BACKBONES:
        print(f"\n--- Testing Backbone: {backbone} ---")
        backbone_results = test_backbone(backbone)
        all_results.update(backbone_results)
        for k, v in backbone_results.items():
            print(f"  {v}  {k}")
    
    # Summary
    print("\n" + "=" * 70)
    n_pass = sum(1 for v in all_results.values() if "PASS" in v)
    n_fail = sum(1 for v in all_results.values() if "FAIL" in v)
    print(f"SUMMARY: {n_pass} passed, {n_fail} failed, {len(all_results)} total")
    
    if n_fail > 0:
        print("\nFAILED TESTS:")
        for k, v in all_results.items():
            if "FAIL" in v:
                print(f"  {v}  {k}")
    
    print("=" * 70)
    return n_fail == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
