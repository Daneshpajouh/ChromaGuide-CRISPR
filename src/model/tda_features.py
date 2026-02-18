"""Topological Data Analysis (TDA) feature extraction wrapper (prototype).

This module provides a small adapter that computes persistence-based
statistics when `ripser` or `giotto-tda` is available; otherwise it falls
back to simple summary statistics so pipelines remain runnable.
"""
import numpy as np
import torch

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except Exception:
    ripser = None
    RIPSER_AVAILABLE = False


def compute_persistence_stats(signal: np.ndarray):
    """Compute simple persistence statistics (lifetimes) from a 1D signal.

    Returns a small vector of summary statistics.
    """
    if RIPSER_AVAILABLE:
        try:
            diagrams = ripser(signal.reshape(-1, 1))['dgms']
            # Flatten lifetimes from H0 and H1
            feats = []
            for d in diagrams:
                lifetimes = d[:, 1] - d[:, 0]
                feats.append(np.nanmean(lifetimes) if len(lifetimes) else 0.0)
                feats.append(np.nanstd(lifetimes) if len(lifetimes) else 0.0)
            return np.array(feats, dtype=float)
        except Exception:
            pass

    # Fallback: simple statistics
    return np.array([np.mean(signal), np.std(signal), np.min(signal), np.max(signal)], dtype=float)


def tda_features_from_tensor(x: torch.Tensor):
    """Accepts a (L,) or (B, L) tensor and returns a numpy array of features.
    """
    if x.ndim == 2:
        # batch
        feats = [compute_persistence_stats(arr.cpu().numpy()) for arr in x]
        return np.stack(feats, axis=0)
    else:
        return compute_persistence_stats(x.cpu().numpy())
