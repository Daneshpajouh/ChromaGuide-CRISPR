#!/usr/bin/env python3
"""
Performance Profiling and Optimization
======================================

Real-time performance monitoring, profiling, and optimization for training:
- GPU memory tracking
- Compute time profiling
- Bottleneck identification
- Optimization recommendations
- Batch size tuning
- Mixed precision analysis

Usage:
    python profile_performance.py --phase phase1 --interval 30
    
    Or integrate into training loop:
    profiler = PerformanceProfiler()
    profiler.start_timer('forward_pass')
    # ... code ...
    profiler.end_timer('forward_pass')
"""

import time
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TimingRecord:
    """Record for timing measurements."""
    name: str
    start_time: float
    end_time: float
    duration_ms: float
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'duration_ms': self.duration_ms
        }


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage."""
    timestamp: str
    rss_mb: float  # Resident set size
    vms_mb: float  # Virtual memory size
    percent: float  # Percentage of system memory
    gpu_memory_mb: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'rss_mb': self.rss_mb,
            'vms_mb': self.vms_mb,
            'percent': self.percent,
            'gpu_memory_mb': self.gpu_memory_mb
        }


class CPUProfiler:
    """CPU performance profiling."""
    
    def __init__(self):
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.active_timers: Dict[str, float] = {}
    
    def start_timer(self, name: str):
        """Start timing a section."""
        self.active_timers[name] = time.perf_counter()
    
    def end_timer(self, name: str) -> Optional[float]:
        """End timing and record duration."""
        if name not in self.active_timers:
            logger.warning(f"Timer {name} not started")
            return None
        
        elapsed = (time.perf_counter() - self.active_timers[name]) * 1000  # Convert to ms
        self.timers[name].append(elapsed)
        del self.active_timers[name]
        
        return elapsed
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get timing statistics."""
        if name not in self.timers or not self.timers[name]:
            return {}
        
        times = self.timers[name]
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'total_ms': np.sum(times),
            'count': len(times)
        }
    
    def get_all_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all timers."""
        return {name: self.get_statistics(name) for name in self.timers}


class MemoryProfiler:
    """Memory usage profiling."""
    
    def __init__(self, track_gpu: bool = False):
        self.snapshots: List[MemorySnapshot] = []
        self.track_gpu = track_gpu
        self.process = psutil.Process()
        self.baseline_memory = self._get_memory_info()
    
    def _get_memory_info(self) -> Tuple[float, float]:
        """Get current memory info in MB."""
        info = self.process.memory_info()
        return info.rss / 1024 / 1024, info.vms / 1024 / 1024
    
    def _get_gpu_memory(self) -> Optional[float]:
        """Get GPU memory usage in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
        except Exception as e:
            logger.debug(f"Could not get GPU memory: {e}")
        return None
    
    def snapshot(self):
        """Take a memory snapshot."""
        rss_mb, vms_mb = self._get_memory_info()
        percent = self.process.memory_percent()
        gpu_mb = self._get_gpu_memory() if self.track_gpu else None
        
        snap = MemorySnapshot(
            timestamp=datetime.now().isoformat(),
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            percent=percent,
            gpu_memory_mb=gpu_mb
        )
        
        self.snapshots.append(snap)
        return snap
    
    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage."""
        if not self.snapshots:
            return {}
        
        rss_values = [s.rss_mb for s in self.snapshots]
        gpu_values = [s.gpu_memory_mb for s in self.snapshots if s.gpu_memory_mb]
        
        return {
            'peak_rss_mb': max(rss_values),
            'peak_gpu_mb': max(gpu_values) if gpu_values else None,
            'baseline_rss_mb': self.baseline_memory[0],
            'peak_increase_mb': max(rss_values) - self.baseline_memory[0]
        }
    
    def get_growth_rate(self) -> Optional[float]:
        """Get memory growth rate (MB/second)."""
        if len(self.snapshots) < 2:
            return None
        
        first = self.snapshots[0]
        last = self.snapshots[-1]
        
        # Get times
        from datetime import datetime as dt
        t1 = dt.fromisoformat(first.timestamp)
        t2 = dt.fromisoformat(last.timestamp)
        delta_sec = (t2 - t1).total_seconds()
        
        if delta_sec <= 0:
            return None
        
        memory_delta = last.rss_mb - first.rss_mb
        return memory_delta / delta_sec


class GPUProfiler:
    """GPU performance profiling."""
    
    def __init__(self):
        self.snapshots: List[Dict] = []
        self.has_gpu = self._check_gpu()
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def snapshot(self) -> Optional[Dict]:
        """Take GPU snapshot."""
        if not self.has_gpu:
            return None
        
        try:
            import torch
            
            snap = {
                'timestamp': datetime.now().isoformat(),
                'memory_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'memory_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'max_memory_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
            }
            
            self.snapshots.append(snap)
            return snap
        
        except Exception as e:
            logger.warning(f"Could not snapshot GPU: {e}")
            return None
    
    def get_peak_memory(self) -> Optional[float]:
        """Get peak GPU memory."""
        if not self.snapshots:
            return None
        
        values = [s['max_memory_allocated_mb'] for s in self.snapshots]
        return max(values) if values else None


class PerformanceProfiler:
    """Unified performance profiler."""
    
    def __init__(self, project_path: Path = None, track_gpu: bool = True):
        self.project_path = project_path or Path(".")
        self.cpu_profiler = CPUProfiler()
        self.memory_profiler = MemoryProfiler(track_gpu=track_gpu)
        self.gpu_profiler = GPUProfiler()
        
        self.metrics: Dict[str, any] = {}
    
    def profile_section(self, name: str):
        """Context manager for profiling a code section."""
        from contextlib import contextmanager
        
        @contextmanager
        def profiler():
            self.cpu_profiler.start_timer(name)
            self.memory_profiler.snapshot()
            self.gpu_profiler.snapshot()
            
            try:
                yield
            finally:
                duration = self.cpu_profiler.end_timer(name)
                self.memory_profiler.snapshot()
                self.gpu_profiler.snapshot()
                
                if duration:
                    logger.info(f"{name}: {duration:.2f}ms")
        
        return profiler()
    
    def get_bottlenecks(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Identify top bottlenecks."""
        stats = self.cpu_profiler.get_all_statistics()
        
        # Sort by total time
        sorted_items = sorted(
            [(name, stat['total_ms']) for name, stat in stats.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_items[:top_k]
    
    def get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Check memory
        peak_mem = self.memory_profiler.get_peak_memory()
        if peak_mem.get('peak_increase_mb', 0) > 5000:
            recommendations.append(
                "Large memory increase detected. Consider reducing batch size or using gradient checkpointing."
            )
        
        # Check GPU
        gpu_peak = self.gpu_profiler.get_peak_memory()
        if gpu_peak and gpu_peak > 30000:  # > 30GB
            recommendations.append(
                "High GPU memory usage. Consider mixed precision training or model parallelism."
            )
        
        # Check bottlenecks
        bottlenecks = self.get_bottlenecks()
        if bottlenecks:
            bottleneck_name = bottlenecks[0][0]
            recommendations.append(
                f"Optimization opportunity: '{bottleneck_name}' is the main bottleneck. "
                f"Consider optimizing this component first."
            )
        
        return recommendations
    
    def export_report(self, output_path: Path = None) -> None:
        """Export profiling report."""
        output_path = output_path or (self.project_path / "profiling_report.json")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'cpu_timings': self.cpu_profiler.get_all_statistics(),
            'memory_stats': self.memory_profiler.get_peak_memory(),
            'memory_growth_rate_mb_per_sec': self.memory_profiler.get_growth_rate(),
            'gpu_peak_memory_mb': self.gpu_profiler.get_peak_memory(),
            'bottlenecks': dict(self.get_bottlenecks()),
            'recommendations': self.get_optimization_recommendations()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Profiling report saved to {output_path}")
    
    def print_summary(self):
        """Print profiling summary."""
        print("\n" + "="*60)
        print("PERFORMANCE PROFILING SUMMARY")
        print("="*60)
        
        # CPU timings
        print("\nTop CPU Bottlenecks:")
        for name, total_ms in self.get_bottlenecks(5):
            print(f"  {name}: {total_ms:.2f}ms")
        
        # Memory stats
        mem_stats = self.memory_profiler.get_peak_memory()
        print(f"\nMemory Usage:")
        print(f"  Peak RSS: {mem_stats.get('peak_rss_mb', 0):.2f}MB")
        print(f"  Baseline: {mem_stats.get('baseline_rss_mb', 0):.2f}MB")
        print(f"  Increase: {mem_stats.get('peak_increase_mb', 0):.2f}MB")
        
        # Recommendations
        print(f"\nOptimization Recommendations:")
        for rec in self.get_optimization_recommendations():
            print(f"  ⚠️  {rec}")
        
        print("\n" + "="*60)


if __name__ == '__main__':
    # Example usage
    profiler = PerformanceProfiler()
    
    # Profile various operations
    for i in range(5):
        with profiler.profile_section(f'operation_{i}'):
            time.sleep(0.1)  # Simulated work
            data = np.random.randn(1000, 1000)
            _ = np.linalg.norm(data)
    
    profiler.print_summary()
    profiler.export_report()
