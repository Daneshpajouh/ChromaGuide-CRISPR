"""
Advanced GPU memory profiling and optimization.

Features:
- GPU memory tracking
- Memory leak detection
- Optimization recommendations
- CUDA profiling
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging


@dataclass
class GPUMemorySnapshot:
    """GPU memory state at point in time."""
    timestamp: float
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    total_mb: float


class GPUMemoryProfiler:
    """Advanced GPU memory profiling."""
    
    def __init__(self):
        self.snapshots: List[GPUMemorySnapshot] = []
        self.peak_memory = 0
        self.memory_history = []
    
    def take_snapshot(self) -> Optional[GPUMemorySnapshot]:
        """Take GPU memory snapshot."""
        try:
            import torch
            import time
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                reserved = torch.cuda.memory_reserved() / 1024 / 1024
                total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                free = total - allocated
                
                snapshot = GPUMemorySnapshot(
                    timestamp=time.time(),
                    allocated_mb=allocated,
                    reserved_mb=reserved,
                    free_mb=free,
                    total_mb=total
                )
                
                self.snapshots.append(snapshot)
                self.peak_memory = max(self.peak_memory, allocated)
                self.memory_history.append(allocated)
                
                return snapshot
        except ImportError:
            logging.warning("PyTorch not available for GPU profiling")
        
        return None
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics."""
        if not self.snapshots:
            return {}
        
        allocated = [s.allocated_mb for s in self.snapshots]
        
        return {
            'mean_allocated_mb': np.mean(allocated),
            'peak_memory_mb': self.peak_memory,
            'min_allocated_mb': np.min(allocated),
            'max_allocated_mb': np.max(allocated),
            'current_allocated_mb': allocated[-1] if allocated else 0
        }
    
    def detect_memory_leak(self, threshold_percent: float = 10.0) -> bool:
        """Detect potential memory leak."""
        if len(self.memory_history) < 10:
            return False
        
        early_avg = np.mean(self.memory_history[:5])
        late_avg = np.mean(self.memory_history[-5:])
        
        increase_percent = ((late_avg - early_avg) / early_avg) * 100
        
        return increase_percent > threshold_percent
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        
        stats = self.get_memory_stats()
        
        if stats['peak_memory_mb'] > 8000:
            recommendations.append("Consider using gradient checkpointing")
            recommendations.append("Use mixed precision (float16) training")
        
        if self.detect_memory_leak():
            recommendations.append("Potential memory leak detected - check data loading")
            recommendations.append("Ensure backward passes are freed after optimizer step")
        
        if stats['peak_memory_mb'] > 10000:
            recommendations.append("Consider smaller batch size")
            recommendations.append("Use distributed training across multiple GPUs")
        
        return recommendations


class CUDAProfiler:
    """CUDA kernel and operation profiling."""
    
    def __init__(self):
        self.operations = []
    
    def profile_forward_pass(self, model, input_shape: Tuple) -> Dict:
        """Profile model forward pass."""
        try:
            import torch
            
            device = next(model.parameters()).device
            x = torch.randn(*input_shape, device=device)
            
            # Simple timing
            import time
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                output = model(x)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            return {
                'forward_time_ms': elapsed * 1000,
                'output_shape': output.shape
            }
        except Exception as e:
            logging.error(f"Forward profiling failed: {e}")
            return {}
    
    def profile_backward_pass(self, model, input_shape: Tuple) -> Dict:
        """Profile model backward pass."""
        try:
            import torch
            
            device = next(model.parameters()).device
            x = torch.randn(*input_shape, device=device, requires_grad=True)
            
            import time
            torch.cuda.synchronize()
            start = time.time()
            
            output = model(x)
            loss = output.mean()
            loss.backward()
            
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            return {
                'backward_time_ms': elapsed * 1000
            }
        except Exception as e:
            logging.error(f"Backward profiling failed: {e}")
            return {}


class XMLProgram:
    """Model extraction and analysis for optimization."""
    
    @staticmethod
    def analyze_model_size(model) -> Dict:
        """Analyze model parameter sizes."""
        total_params = 0
        param_sizes = {}
        
        for name, param in model.named_parameters():
            size = param.numel()
            total_params += size
            param_sizes[name] = {
                'num_params': size,
                'param_shape': tuple(param.shape),
                'memory_mb': (size * 4) / 1024 / 1024  # float32
            }
        
        return {
            'total_params': total_params,
            'total_memory_mb': (total_params * 4) / 1024 / 1024,
            'layer_details': param_sizes
        }
    
    @staticmethod
    def get_model_flops(model, input_shape: Tuple) -> float:
        """Estimate FLOPs for forward pass."""
        try:
            from thop import profile
            import torch
            
            device = next(model.parameters()).device
            x = torch.randn(*input_shape, device=device)
            flops, _ = profile(model, inputs=(x,))
            return flops / 1e9  # Convert to GFLOPs
        except:
            return 0


class GPUUtilizationOptimizer:
    """Optimize GPU utilization."""
    
    @staticmethod
    def find_optimal_batch_size(
        model,
        memory_limit_mb: int = 8000,
        input_shape: Optional[Tuple] = None
    ) -> int:
        """Find optimal batch size for GPU memory."""
        import torch
        
        if input_shape is None:
            input_shape = (1, 3, 224, 224)
        
        for batch_size in [2**i for i in range(1, 12)]:  # 2, 4, 8, 16, ..., 2048
            test_shape = (batch_size,) + input_shape[1:]
            
            try:
                x = torch.randn(*test_shape, device='cuda')
                with torch.no_grad():
                    _ = model(x)
                torch.cuda.empty_cache()
            except RuntimeError:
                return max(1, batch_size // 2)
        
        return 1024  # Default if no limit found
    
    @staticmethod
    def enable_mixed_precision(model) -> torch.nn.Module:
        """Enable automatic mixed precision."""
        try:
            import torch
            from torch.cuda.amp import autocast
            
            # This is a wrapper
            class AutocastModel(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    with autocast():
                        return self.model(x)
            
            return AutocastModel(model)
        except:
            return model
