#!/usr/bin/env python3
"""
Comprehensive Logging and Analytics System
===========================================

Advanced logging, metrics collection, and analytics for the pipeline:
- Structured logging with multiple handlers
- Real-time metrics collection
- Performance monitoring
- Alert system
- Analytics dashboard data pipeline

Usage:
    from analytics import PipelineAnalytics
    
    analytics = PipelineAnalytics()
    analytics.log_phase_start('phase1', {'job_id': '56644478'})
    analytics.log_metric('loss', 0.125, phase='phase1', epoch=5)
    analytics.log_event('gpu_allocated', severity='info')
"""

import logging
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import sqlite3
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
from collections import defaultdict


class Severity(Enum):
    """Log severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """Metric types."""
    SCALAR = "scalar"
    HISTOGRAM = "histogram"
    COUNTER = "counter"
    GAUGE = "gauge"
    TIMING = "timing"


@dataclass
class LogEvent:
    """Single log event."""
    timestamp: str
    phase: str
    event_type: str
    severity: str
    message: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Metric:
    """Single metric measurement."""
    timestamp: str
    phase: str
    metric_name: str
    value: float
    metric_type: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class MetricsCollector:
    """Collect and aggregate metrics."""
    
    def __init__(self):
        self.metrics: List[Metric] = []
        self.metric_aggregates = defaultdict(list)
    
    def record_metric(self, phase: str, name: str, value: float,
                     metric_type: str = "scalar", metadata: Optional[Dict] = None):
        """Record a single metric."""
        metric = Metric(
            timestamp=datetime.now().isoformat(),
            phase=phase,
            metric_name=name,
            value=value,
            metric_type=metric_type,
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        self.metric_aggregates[f"{phase}:{name}"].append(value)
    
    def get_statistics(self, phase: str, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        key = f"{phase}:{metric_name}"
        if key not in self.metric_aggregates:
            return {}
        
        values = self.metric_aggregates[key]
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'count': len(values),
            'sum': np.sum(values)
        }
    
    def export_metrics(self, filepath: Path) -> None:
        """Export metrics to JSON."""
        data = [m.to_dict() for m in self.metrics]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class PipelineLogger:
    """Structured logging for pipeline."""
    
    def __init__(self, log_dir: Path = None):
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        self.events: List[LogEvent] = []
        
        # Configure loggers for different outputs
        self._setup_file_logging()
        self._setup_console_logging()
    
    def _setup_file_logging(self):
        """Setup file logging."""
        self.file_logger = logging.getLogger('pipeline_file')
        handler = logging.FileHandler(self.log_dir / 'pipeline.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.file_logger.addHandler(handler)
        self.file_logger.setLevel(logging.DEBUG)
    
    def _setup_console_logging(self):
        """Setup console logging."""
        self.console_logger = logging.getLogger('pipeline_console')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        self.console_logger.addHandler(handler)
        self.console_logger.setLevel(logging.INFO)
    
    def log_event(self, phase: str, event_type: str, message: str,
                 severity: str = "info", metadata: Optional[Dict] = None):
        """Log an event."""
        event = LogEvent(
            timestamp=datetime.now().isoformat(),
            phase=phase,
            event_type=event_type,
            severity=severity.upper(),
            message=message,
            metadata=metadata or {}
        )
        
        self.events.append(event)
        
        # Also log via standard logger
        log_method = getattr(self.file_logger, severity.lower(), self.file_logger.info)
        log_method(f"[{phase}] {event_type}: {message}")
        
        # Console output for important events
        if severity.upper() in ['ERROR', 'CRITICAL', 'WARNING']:
            console_method = getattr(self.console_logger, severity.lower(),
                                    self.console_logger.info)
            console_method(f"[{phase}] {event_type}: {message}")
    
    def export_events(self, filepath: Path) -> None:
        """Export events to JSON."""
        data = [e.to_dict() for e in self.events]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class AlertSystem:
    """Alert system for anomalies and important events."""
    
    def __init__(self):
        self.alerts: List[Dict] = []
        self.alert_rules = {
            'loss_diverging': False,
            'gpu_memory_low': False,
            'training_slow': False,
            'validation_worse': False
        }
    
    def check_loss_divergence(self, loss: float, prev_loss: float, threshold: float = 1.5) -> bool:
        """Check if loss is diverging."""
        if prev_loss > 0 and loss > prev_loss * threshold:
            return True
        return False
    
    def check_gpu_memory(self, used_mb: float, total_mb: float, threshold: float = 0.9) -> bool:
        """Check if GPU memory is running low."""
        return (used_mb / total_mb) > threshold
    
    def check_convergence(self, metrics_history: List[float], min_improvement: float = 0.001) -> bool:
        """Check if model is converging slowly."""
        if len(metrics_history) < 5:
            return False
        
        recent = metrics_history[-5:]
        improvement = (recent[-1] - recent[0]) / abs(recent[0])
        return improvement < min_improvement
    
    def raise_alert(self, alert_type: str, severity: str, message: str,
                   metadata: Optional[Dict] = None):
        """Raise an alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'severity': severity,
            'message': message,
            'metadata': metadata or {}
        }
        
        self.alerts.append(alert)
        
        # Log alert
        severity_map = {
            'critical': logging.CRITICAL,
            'high': logging.ERROR,
            'medium': logging.WARNING,
            'low': logging.INFO
        }
        
        logger = logging.getLogger('alerts')
        log_level = severity_map.get(severity, logging.INFO)
        logger.log(log_level, f"[{alert_type}] {message}")


class PipelineAnalytics:
    """Main analytics coordinator."""
    
    def __init__(self, project_path: Path = None):
        self.project_path = project_path or Path(".")
        self.logger = PipelineLogger(self.project_path / "logs")
        self.metrics = MetricsCollector()
        self.alerts = AlertSystem()
        
        # Database setup
        self.db_path = self.project_path / "pipeline_analytics.db"
        self._setup_database()
        
        # Timing tracking
        self.phase_start_times = {}
    
    def _setup_database(self):
        """Setup SQLite database for analytics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                phase TEXT,
                event_type TEXT,
                severity TEXT,
                message TEXT,
                metadata TEXT
            )
        ''')
        
        # Metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                phase TEXT,
                metric_name TEXT,
                value REAL,
                metric_type TEXT,
                metadata TEXT
            )
        ''')
        
        # Phases table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS phases (
                id INTEGER PRIMARY KEY,
                phase_name TEXT UNIQUE,
                start_time TEXT,
                end_time TEXT,
                duration_seconds REAL,
                status TEXT,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_phase_start(self, phase: str, metadata: Optional[Dict] = None):
        """Log phase start."""
        self.phase_start_times[phase] = time.time()
        self.logger.log_event(
            phase=phase,
            event_type='phase_start',
            message=f"Starting {phase}",
            severity='info',
            metadata=metadata
        )
    
    def log_phase_end(self, phase: str, status: str = "success", metadata: Optional[Dict] = None):
        """Log phase end."""
        duration = None
        if phase in self.phase_start_times:
            duration = time.time() - self.phase_start_times[phase]
        
        self.logger.log_event(
            phase=phase,
            event_type='phase_end',
            message=f"Completed {phase} with status {status}",
            severity='info',
            metadata={**(metadata or {}), 'duration_seconds': duration}
        )
    
    def log_metric(self, metric_name: str, value: float, phase: str = "general",
                  metric_type: str = "scalar", metadata: Optional[Dict] = None):
        """Log a metric."""
        self.metrics.record_metric(phase, metric_name, value, metric_type, metadata)
    
    def log_event(self, event_type: str, message: str, phase: str = "general",
                 severity: str = "info", metadata: Optional[Dict] = None):
        """Log an event."""
        self.logger.log_event(phase, event_type, message, severity, metadata)
    
    def export_analytics(self, output_dir: Path = None):
        """Export all analytics."""
        output_dir = output_dir or (self.project_path / "analytics")
        output_dir.mkdir(exist_ok=True)
        
        # Export events
        self.logger.export_events(output_dir / "events.json")
        
        # Export metrics
        self.metrics.export_metrics(output_dir / "metrics.json")
        
        # Export summary
        summary = {
            'total_events': len(self.logger.events),
            'total_metrics': len(self.metrics.metrics),
            'total_alerts': len(self.alerts.alerts),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def get_phase_summary(self, phase: str) -> Dict[str, Any]:
        """Get summary statistics for a phase."""
        phase_events = [e for e in self.logger.events if e.phase == phase]
        phase_metrics = [m for m in self.metrics.metrics if m.phase == phase]
        
        return {
            'phase': phase,
            'event_count': len(phase_events),
            'metric_count': len(phase_metrics),
            'events': [e.to_dict() for e in phase_events],
            'metrics': [m.to_dict() for m in phase_metrics]
        }


# Global analytics instance
_analytics_instance = None

def get_analytics(project_path: Path = None) -> PipelineAnalytics:
    """Get or create global analytics instance."""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = PipelineAnalytics(project_path)
    return _analytics_instance


if __name__ == '__main__':
    # Example usage
    analytics = PipelineAnalytics()
    
    # Log phase
    analytics.log_phase_start('phase1', {'job_id': '56644478', 'host': 'narval'})
    
    # Log metrics
    for epoch in range(10):
        analytics.log_metric('loss', 0.5 - epoch * 0.03, phase='phase1',
                            metadata={'epoch': epoch})
        analytics.log_metric('spearman_r', 0.5 + epoch * 0.03, phase='phase1',
                            metadata={'epoch': epoch})
    
    analytics.log_phase_end('phase1', status='success')
    
    # Export
    analytics.export_analytics()
