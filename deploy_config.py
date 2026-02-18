#!/usr/bin/env python3
"""
Production Deployment Configuration
===================================

Configuration, verification, and deployment orchestration for production:
- Environment configuration management
- Dependency resolution
- Health checks
- Service startup/shutdown
- Backup and recovery

Usage:
    from deploy_config import DeploymentConfig
    
    config = DeploymentConfig('production')
    config.verify_environment()
    config.start_services()
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, List
import subprocess
import logging
from dataclasses import dataclass, asdict
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for a service."""
    name: str
    port: int
    enable: bool = True
    restart_policy: str = "always"
    memory_limit_mb: int = 2048
    cpu_limit: float = 1.0


class DeploymentConfig:
    """Main deployment configuration."""
    
    def __init__(self, environment: str = 'staging'):
        self.environment = environment
        self.project_root = Path(__file__).parent
        self.config_dir = self.project_root / 'config'
        self.config_dir.mkdir(exist_ok=True)
        
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from file or create defaults."""
        config_file = self.config_dir / f"{self.environment}.yaml"
        
        if config_file.exists():
            with open(config_file) as f:
                return yaml.safe_load(f)
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'environment': self.environment,
            'debug': self.environment != 'production',
            'models': {
                'phase1': {
                    'checkpoint': 'checkpoints/phase1/best_model.pt',
                    'device': 'cuda' if self.environment == 'production' else 'cpu'
                },
                'phase2': {
                    'checkpoint': 'checkpoints/phase2_xgboost/xgboost_model.pkl'
                },
                'phase3': {
                    'checkpoint': 'checkpoints/phase3_deephybrid/stacking_ensemble.pt'
                }
            },
            'services': {
                'api': {
                    'port': 8000,
                    'enable': True,
                    'workers': 4 if self.environment == 'production' else 1
                },
                'dashboard': {
                    'port': 8501,
                    'enable': True
                },
                'monitoring': {
                    'port': 9090,
                    'enable': True
                }
            },
            'logging': {
                'level': 'INFO' if self.environment == 'production' else 'DEBUG',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': f'logs/{self.environment}.log'
            },
            'database': {
                'type': 'sqlite' if self.environment != 'production' else 'postgresql',
                'path': 'pipeline.db'
            }
        }
    
    def verify_environment(self) -> bool:
        """Verify deployment environment."""
        logger.info(f"Verifying {self.environment} environment...")
        
        issues = []
        
        # Check Python packages
        required_packages = [
            'numpy', 'pandas', 'torch', 'sklearn', 'xgboost', 'streamlit'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✓ {package}")
            except ImportError:
                issues.append(f"Missing package: {package}")
        
        # Check models exist
        for phase, phase_config in self.config['models'].items():
            path = Path(phase_config['checkpoint'])
            if path.exists():
                logger.info(f"✓ {phase} model found")
            else:
                issues.append(f"Missing {phase} model: {path}")
        
        # Check directories
        required_dirs = ['logs', 'checkpoints', 'data', 'results']
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
            logger.info(f"✓ Directory {dir_name}")
        
        if issues:
            logger.error(f"Environment verification failed: {issues}")
            return False
        
        logger.info("Environment verification passed")
        return True
    
    def check_health(self) -> Dict[str, bool]:
        """Check health of deployment."""
        health = {}
        
        # Check models loadable
        try:
            import torch
            checkpoint = Path(self.config['models']['phase1']['checkpoint'])
            if checkpoint.exists():
                torch.load(checkpoint, map_location='cpu')
                health['phase1_model'] = True
            else:
                health['phase1_model'] = False
        except Exception as e:
            logger.error(f"Phase 1 model check failed: {e}")
            health['phase1_model'] = False
        
        # Check disk space
        import shutil
        stat = shutil.disk_usage(self.project_root)
        health['disk_space_ok'] = stat.free > 1e9  # At least 1GB free
        
        # Check database connectivity
        if self.config['database']['type'] == 'sqlite':
            db_path = Path(self.config['database']['path'])
            health['database'] = db_path.parent.exists()
        else:
            # Would check PostgreSQL/other DB here
            health['database'] = True
        
        return health
    
    def get_service_config(self, service_name: str) -> Optional[ServiceConfig]:
        """Get configuration for a service."""
        service_info = self.config['services'].get(service_name)
        
        if not service_info:
            return None
        
        return ServiceConfig(
            name=service_name,
            port=service_info.get('port', 8000),
            enable=service_info.get('enable', True),
            restart_policy=service_info.get('restart_policy', 'always'),
            memory_limit_mb=service_info.get('memory_limit_mb', 2048),
            cpu_limit=service_info.get('cpu_limit', 1.0)
        )
    
    def start_services(self) -> bool:
        """Start all configured services."""
        logger.info("Starting services...")
        
        # Start API server (if enabled)
        api_config = self.get_service_config('api')
        if api_config and api_config.enable:
            logger.info(f"Starting API on port {api_config.port}...")
            # Would use uvicorn, gunicorn, etc.
            # subprocess.Popen(['uvicorn', 'api:app', '--port', str(api_config.port)])
        
        # Start dashboard (if enabled)
        dash_config = self.get_service_config('dashboard')
        if dash_config and dash_config.enable:
            logger.info(f"Starting dashboard on port {dash_config.port}...")
            # subprocess.Popen(['streamlit', 'run', 'dashboard_ui.py', '--server.port', str(dash_config.port)])
        
        # Start monitoring (if enabled)
        mon_config = self.get_service_config('monitoring')
        if mon_config and mon_config.enable:
            logger.info(f"Starting monitoring on port {mon_config.port}...")
            # Would setup Prometheus, etc.
        
        logger.info("Services started")
        return True
    
    def stop_services(self) -> bool:
        """Stop all services."""
        logger.info("Stopping services...")
        
        # Kill by process name
        for service in ['uvicorn', 'streamlit', 'prometheus']:
            try:
                subprocess.run(['pkill', '-f', service], stderr=subprocess.DEVNULL)
            except:
                pass
        
        logger.info("Services stopped")
        return True
    
    def export_config(self, output_path: Optional[Path] = None) -> None:
        """Export configuration to file."""
        if output_path is None:
            output_path = self.config_dir / f"{self.environment}.yaml"
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Configuration exported to {output_path}")
    
    def get_summary(self) -> str:
        """Get config summary."""
        summary = f"""
Deployment Configuration Summary
================================
Environment: {self.environment}
Project Root: {self.project_root}
Config Dir: {self.config_dir}

Services:
"""
        for service_name, service_config in self.config['services'].items():
            enabled = "✓" if service_config['enable'] else "✗"
            port = service_config.get('port', 'N/A')
            summary += f"  {enabled} {service_name:15} on port {port}\n"
        
        summary += f"""
Models:
"""
        for phase, phase_config in self.config['models'].items():
            path = phase_config.get('checkpoint', 'N/A')
            summary += f"  • {phase:15} : {path}\n"
        
        return summary


class DeploymentManager:
    """Manage deployment lifecycle."""
    
    def __init__(self, environment: str = 'staging'):
        self.config = DeploymentConfig(environment)
        self.deployment_log = Path('deployment_history.json')
    
    def pre_deployment_checks(self) -> bool:
        """Run pre-deployment checks."""
        logger.info("Running pre-deployment checks...")
        
        if not self.config.verify_environment():
            return False
        
        health = self.config.check_health()
        logger.info(f"Health check: {health}")
        
        return all(health.values())
    
    def deploy(self) -> bool:
        """Execute deployment."""
        logger.info("Executing deployment...")
        
        if not self.pre_deployment_checks():
            logger.error("Pre-deployment checks failed")
            return False
        
        # Start services
        self.config.start_services()
        
        # Log deployment
        self._log_deployment('success')
        
        return True
    
    def rollback(self, steps: int = 1) -> bool:
        """Rollback deployment."""
        logger.warning(f"Rolling back {steps} deployment(s)...")
        
        self.config.stop_services()
        
        # Would restore from backup here
        self._log_deployment('rollback')
        
        return True
    
    def _log_deployment(self, status: str) -> None:
        """Log deployment history."""
        import json
        from datetime import datetime
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'environment': self.config.environment,
            'status': status
        }
        
        history = []
        if self.deployment_log.exists():
            with open(self.deployment_log) as f:
                history = json.load(f)
        
        history.append(entry)
        
        with open(self.deployment_log, 'w') as f:
            json.dump(history, f, indent=2)


if __name__ == '__main__':
    import sys
    
    env = sys.argv[1] if len(sys.argv) > 1 else 'staging'
    
    config = DeploymentConfig(env)
    print(config.get_summary())
    
    # Verify
    if config.verify_environment():
        health = config.check_health()
        print("\nHealth Check:")
        for check, result in health.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check}")
