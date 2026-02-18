"""ExperimentLogger: structured experiment logging for training runs.

Saves timestamped JSON (and YAML if available) to `logs/` and provides
per-epoch logging hooks. Gathers hardware info, hyperparameters, model
summary (best-effort) and runtime statistics.
"""
from __future__ import annotations
import json
import os
import time
import platform
from datetime import datetime
from typing import Any, Dict, Optional
import logging

try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    yaml = None
    YAML_AVAILABLE = False

LOG = logging.getLogger(__name__)


def _timestamp():
    return datetime.utcnow().isoformat() + 'Z'


class ExperimentLogger:
    def __init__(self, run_name: Optional[str] = None, out_dir: str = 'logs'):
        self.run_name = run_name or datetime.utcnow().strftime('run_%Y%m%dT%H%M%SZ')
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.start_time = time.time()
        self.epochs = []
        self.meta: Dict[str, Any] = {}
        self.hardware = self._gather_hardware_info()

    def _gather_hardware_info(self) -> Dict[str, Any]:
        info = {
            'platform': platform.platform(),
            'system': platform.system(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
        }
        if PSUTIL_AVAILABLE:
            try:
                info.update({
                    'cpu_count': psutil.cpu_count(logical=True),
                    'mem_total_bytes': psutil.virtual_memory().total,
                })
            except Exception:
                pass
        return info

    def start_run(self, hyperparams: Dict[str, Any], model: Optional[Any] = None, dataset_info: Optional[Dict[str, Any]] = None):
        self.meta['run_name'] = self.run_name
        self.meta['started_at'] = _timestamp()
        self.meta['hyperparameters'] = hyperparams
        self.meta['dataset'] = dataset_info or {}
        self.meta['hardware'] = self.hardware
        # best-effort model summary
        try:
            if model is not None:
                # try parameter count
                params = sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else None
                self.meta['model_summary'] = {'param_count': int(params) if params is not None else None, 'repr': repr(model)[:1000]}
        except Exception:
            pass

    def log_epoch(self, epoch: int, duration_s: float, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any], lr: Optional[float] = None, grad_norm: Optional[float] = None):
        entry = {
            'epoch': int(epoch),
            'timestamp': _timestamp(),
            'duration_s': float(duration_s),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'lr': lr,
            'grad_norm': grad_norm,
        }
        self.epochs.append(entry)
        # incremental flush
        self._flush_partial()

    def _flush_partial(self):
        path = os.path.join(self.out_dir, f"{self.run_name}.partial.json")
        try:
            with open(path, 'w') as fh:
                json.dump({'meta': self.meta, 'epochs': self.epochs}, fh, indent=2)
        except Exception as e:
            LOG.warning(f"Failed to flush partial experiment log: {e}")

    def finalize(self, best_info: Optional[Dict[str, Any]] = None, checkpoints: Optional[Dict[str, Any]] = None):
        self.meta['finished_at'] = _timestamp()
        self.meta['duration_s'] = time.time() - self.start_time
        if checkpoints is not None:
            self.meta['checkpoints'] = checkpoints
        if best_info is not None:
            self.meta['best'] = best_info
        out = {'meta': self.meta, 'epochs': self.epochs}

        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        json_path = os.path.join(self.out_dir, f"{self.run_name}.{timestamp}.json")
        try:
            with open(json_path, 'w') as fh:
                json.dump(out, fh, indent=2)
            LOG.info(f"Saved experiment log: {json_path}")
        except Exception as e:
            LOG.warning(f"Failed to save final experiment log: {e}")

        if YAML_AVAILABLE:
            try:
                yaml_path = os.path.join(self.out_dir, f"{self.run_name}.{timestamp}.yaml")
                with open(yaml_path, 'w') as fh:
                    yaml.safe_dump(out, fh)
                LOG.info(f"Saved experiment YAML: {yaml_path}")
            except Exception:
                pass

        # print concise summary and a small table of best epoch
        try:
            best_epoch = None
            if self.epochs:
                # choose best by val spearman if present
                best_epoch = max(self.epochs, key=lambda e: e.get('val_metrics', {}).get('spearman', float('-inf')))
            LOG.info(f"Run {self.run_name} finished. Epochs={len(self.epochs)} best_epoch={best_epoch.get('epoch') if best_epoch else None}")
            if best_epoch is not None:
                be = best_epoch
                summary = {
                    'epoch': be.get('epoch'),
                    'val_spearman': be.get('val_metrics', {}).get('spearman'),
                    'val_pearson': be.get('val_metrics', {}).get('pearson'),
                    'val_mse': be.get('val_metrics', {}).get('mse'),
                    'duration_s': be.get('duration_s')
                }
                LOG.info(f"Best epoch summary: {summary}")
        except Exception:
            LOG.info(f"Run {self.run_name} finished. Epochs={len(self.epochs)}")
