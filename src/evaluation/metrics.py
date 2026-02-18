"""Evaluation metrics: Spearman, Cohen's d, AUROC.

This module tries to use scipy/sklearn where available and falls back to
numpy-based implementations if not. The functions accept lists or numpy
arrays and return scalar metrics.
"""
from typing import Sequence
import logging

try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy.stats import spearmanr
    SCIPY_AVAILABLE = True
except Exception:
    spearmanr = None
    SCIPY_AVAILABLE = False

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except Exception:
    roc_auc_score = None
    SKLEARN_AVAILABLE = False
try:
    from sklearn.metrics import average_precision_score, mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_FULL = True
except Exception:
    average_precision_score = None
    mean_squared_error = None
    mean_absolute_error = None
    r2_score = None
    SKLEARN_FULL = False

logger = logging.getLogger(__name__)


def spearman_correlation(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if SCIPY_AVAILABLE:
        rho, _ = spearmanr(y_true, y_pred)
        return float(rho) if rho is not None else 0.0

    # fallback: rank-transform then Pearson
    try:
        y_true = _as_numpy(y_true)
        y_pred = _as_numpy(y_pred)
        def ranks(a):
            return np.argsort(np.argsort(a)).astype(float)
        r1 = ranks(y_true)
        r2 = ranks(y_pred)
        if r1.std() == 0 or r2.std() == 0:
            return 0.0
        rho = ((r1 - r1.mean()) * (r2 - r2.mean())).mean() / (r1.std() * r2.std())
        return float(rho)
    except Exception as e:
        logger.warning(f"Spearman fallback failed: {e}")
        return 0.0


def cohen_d(y1: Sequence[float], y2: Sequence[float]) -> float:
    y1 = _as_numpy(y1)
    y2 = _as_numpy(y2)
    n1 = len(y1)
    n2 = len(y2)
    s1 = y1.std(ddof=1)
    s2 = y2.std(ddof=1)
    # pooled sd
    s = (( (n1-1)*(s1**2) + (n2-1)*(s2**2) ) / (n1 + n2 - 2)) ** 0.5
    if s == 0:
        return 0.0
    d = (y1.mean() - y2.mean()) / s
    return float(d)


def auroc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    if SKLEARN_AVAILABLE:
        try:
            return float(roc_auc_score(y_true, y_score))
        except Exception as e:
            logger.warning(f"sklearn roc_auc_score failed: {e}")

    # fallback: simple ROC AUC using trapezoidal rule
    try:
        y_true = _as_numpy(y_true)
        y_score = _as_numpy(y_score)
        # sort by score desc
        desc = np.argsort(-y_score)
        y_true = y_true[desc]
        # compute tpr/fpr points
        positives = y_true.sum()
        negatives = len(y_true) - positives
        if positives == 0 or negatives == 0:
            return 0.0
        tprs = []
        fprs = []
        tp = 0
        fp = 0
        for val in y_true:
            if val:
                tp += 1
            else:
                fp += 1
            tprs.append(tp / positives)
            fprs.append(fp / negatives)
        # integrate using trapezoidal rule
        tprs = np.array(tprs)
        fprs = np.array(fprs)
        # ensure sorted by fpr
        order = np.argsort(fprs)
        auc = np.trapezoid(tprs[order], fprs[order])
        return float(auc)
    except Exception as e:
        logger.warning(f"AUROC fallback failed: {e}")
        return 0.0


def auprc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Area under Precision-Recall curve (average precision)."""
    if SKLEARN_FULL and average_precision_score is not None:
        try:
            return float(average_precision_score(y_true, y_score))
        except Exception as e:
            logger.warning(f"sklearn average_precision_score failed: {e}")

    # fallback: simple precision-recall integration
    try:
        y_true = _as_numpy(y_true).astype(int)
        y_score = _as_numpy(y_score)
        desc = np.argsort(-y_score)
        y_true = y_true[desc]
        tp = 0
        fp = 0
        precisions = []
        recalls = []
        positives = y_true.sum()
        for i, val in enumerate(y_true, start=1):
            if val:
                tp += 1
            else:
                fp += 1
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            recalls.append(tp / positives if positives > 0 else 0.0)
        # integrate precision w.r.t recall using trapezoid
        if len(recalls) < 2:
            return float(precisions[0]) if precisions else 0.0
        return float(np.trapezoid(precisions, recalls))
    except Exception as e:
        logger.warning(f"AUPRC fallback failed: {e}")
        return 0.0


def pearson_correlation(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    try:
        y_true = _as_numpy(y_true).astype(float)
        y_pred = _as_numpy(y_pred).astype(float)
        if y_true.size == 0 or y_pred.size == 0:
            return 0.0
        # use numpy Pearson
        xm = y_true.mean()
        ym = y_pred.mean()
        xm2 = ((y_true - xm) ** 2).sum()
        ym2 = ((y_pred - ym) ** 2).sum()
        denom = (xm2 * ym2) ** 0.5
        if denom == 0:
            return 0.0
        cov = ((y_true - xm) * (y_pred - ym)).sum()
        return float(cov / denom)
    except Exception as e:
        logger.warning(f"Pearson fallback failed: {e}")
        return 0.0


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if SKLEARN_FULL and mean_squared_error is not None:
        try:
            return float(mean_squared_error(y_true, y_pred))
        except Exception:
            pass
    y_true = _as_numpy(y_true).astype(float)
    y_pred = _as_numpy(y_pred).astype(float)
    return float(((y_true - y_pred) ** 2).mean())


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if SKLEARN_FULL and mean_absolute_error is not None:
        try:
            return float(mean_absolute_error(y_true, y_pred))
        except Exception:
            pass
    y_true = _as_numpy(y_true).astype(float)
    y_pred = _as_numpy(y_pred).astype(float)
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(mse(y_true, y_pred) ** 0.5)


def r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if SKLEARN_FULL and r2_score is not None:
        try:
            return float(r2_score(y_true, y_pred))
        except Exception:
            pass
    y_true = _as_numpy(y_true).astype(float)
    y_pred = _as_numpy(y_pred).astype(float)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_regression_metrics(y_true: Sequence[float], y_pred: Sequence[float], y_scores=None) -> dict:
    """Return a dict with regression metrics only (safe for continuous targets).

    AUROC/AUPRC are classification metrics and are not computed by default.
    """
    out = {}
    out['spearman'] = spearman_correlation(y_true, y_pred)
    out['pearson'] = pearson_correlation(y_true, y_pred)
    out['mse'] = mse(y_true, y_pred)
    out['mae'] = mae(y_true, y_pred)
    out['rmse'] = rmse(y_true, y_pred)
    out['r2'] = r2(y_true, y_pred)
    return out


def _as_numpy(x):
    if np is None:
        raise RuntimeError('numpy is required for metrics fallback')
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.array(x)
