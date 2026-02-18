"""Device detection and utilities for ChromaGuide.

Priority: CUDA > MPS (Apple Silicon) > CPU

Provides:
- get_best_device() -> torch.device
- DeviceManager: wrapper that holds device and helpers to move models/tensors
"""
from typing import Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False


def get_best_device() -> Any:
    """Return the best available torch device (torch.device) with priority CUDA > MPS > CPU.

    If torch is not available, returns the string 'cpu'.
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available; defaulting device to 'cpu'")
        return 'cpu'

    # CUDA has highest priority
    try:
        if torch.cuda.is_available():
            dev = torch.device('cuda')
            logger.info(f"Using CUDA device: {dev}")
            return dev
    except Exception:
        pass

    # Apple Silicon MPS
    try:
        mps = getattr(torch.backends, 'mps', None)
        if mps is not None and mps.is_available():
            dev = torch.device('mps')
            logger.info(f"Using Apple MPS device: {dev}")
            return dev
    except Exception:
        pass

    # Fallback to CPU
    dev = torch.device('cpu')
    logger.info(f"Falling back to CPU device: {dev}")
    return dev


class DeviceManager:
    """Helper to manage device placement for models and tensors.

    Usage:
        dm = DeviceManager()            # auto-detect
        dm = DeviceManager('cuda')      # explicit
        dm.to_device(model)
        x = dm.tensor_to_device(x)
    """

    def __init__(self, device: Optional[Any] = None):
        if device is None:
            self.device = get_best_device() if TORCH_AVAILABLE else 'cpu'
        else:
            # accept strings like 'cuda','mps','cpu' or torch.device
            if TORCH_AVAILABLE and not isinstance(device, str) and isinstance(device, torch.device):
                self.device = device
            else:
                try:
                    self.device = torch.device(device) if TORCH_AVAILABLE else device
                except Exception:
                    self.device = device

        self.device_str = str(self.device)
        logger.info(f"DeviceManager initialized with device={self.device_str}")

    def to_device(self, module: Any):
        """Move a torch.nn.Module or tensor-like object to the manager's device.

        Returns the moved object. If torch is not available, returns the input unchanged.
        """
        if not TORCH_AVAILABLE:
            return module
        try:
            if hasattr(module, 'to'):
                return module.to(self.device)
        except Exception:
            # swallow and return original
            logger.debug("Failed to to() module; returning original")
        return module

    def tensor_to_device(self, x: Any):
        """Move tensors (or nested tuples/lists) to device.

        Supports torch.Tensor, lists, tuples. Returns moved object.
        """
        if not TORCH_AVAILABLE:
            return x
        if torch.is_tensor(x):
            return x.to(self.device)
        if isinstance(x, (list, tuple)):
            return type(x)(self.tensor_to_device(xx) for xx in x)
        return x

    def is_cuda(self) -> bool:
        if not TORCH_AVAILABLE:
            return False
        try:
            return self.device.type == 'cuda'
        except Exception:
            return False

    def is_mps(self) -> bool:
        if not TORCH_AVAILABLE:
            return False
        try:
            return self.device.type == 'mps'
        except Exception:
            return False

    def is_cpu(self) -> bool:
        if not TORCH_AVAILABLE:
            return True
        try:
            return self.device.type == 'cpu'
        except Exception:
            return True


def warn_if_pythonpath():
    """Warn if PYTHONPATH is set — this can interfere with conda/env isolation."""
    py = os.environ.get('PYTHONPATH')
    if py:
        logger.warning(f"PYTHONPATH is set ({py}) — this may cause imports from outside the active environment. Consider unsetting PYTHONPATH when using conda/mamba environments.")
        return True
    return False
