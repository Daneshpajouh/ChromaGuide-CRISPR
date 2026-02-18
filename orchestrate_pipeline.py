#!/usr/bin/env python3
"""
Multi-Phase Autonomous Orchestration Script
============================================

Orchestrates complete end-to-end ChromaGuide research pipeline:
1. Monitor Phase 1 (DNABERT-Mamba) completion on Narval
2. Launch Phase 2 (XGBoost benchmarking)
3. Launch Phase 3 (DeepHybrid ensemble)
4. Launch Phase 4 (Clinical validation)
5. Run comprehensive benchmarking
6. Generate publication figures
7. Update Overleaf with results
8. Commit to GitHub with automatic versioning

Usage:
    python orchestrate_pipeline.py --watch-job 56644478 --auto-push-github

This script can be left running and will automatically execute each phase
as dependencies complete.
"""

import argparse
import logging
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates multi-phase research pipeline."""
    
    PHASES = {
        'phase1': {
            'name': 'DNABERT-Mamba Training',
            'status': 'queued',
            'dependencies': [],
            'location': 'narval',
            'job_id': None,
        },
        'phase2': {
            'name': 'CRISPRO-XGBoost Benchmarking',
            'status': 'waiting',
            'dependencies': ['phase1'],
            'location': 'local',
            'script': 'train_phase2_xgboost.py',
        },
        'phase3': {
            'name': 'DeepHybrid Ensemble',
            'status': 'waiting',
            'dependencies': ['phase1', 'phase2'],
            'location': 'narval',
            'script': 'train_phase3_deephybrid.py',
        },
        'phase4': {
            'name': 'Clinical Validation',
            'status': 'waiting',
            'dependencies': ['phase1', 'phase3'],
            'location': 'local',
            'script': 'train_phase4_clinical_validation.py',
        },
        'benchmarking': {
            'name': 'SOTA Benchmarking',
            'status': 'waiting',
            'dependencies': ['phase3', 'phase4'],
            'location': 'local',
            'script': 'benchmark_sota.py',
        },
        'figures': {
            'name': 'Figure Generation',
            'status': 'waiting',
            'dependencies': ['benchmarking'],
            'location': 'local',
            'script': 'generate_figures.py',
        },
        'overleaf': {
            'name': 'Update Overleaf',
            'status': 'waiting',
            'dependencies': ['figures'],
            'location': 'local',
            'script': 'inject_results_overleaf.py',
        },
        'commit': {
            'name': 'Git Commit & Tag',
            'status': 'waiting',
            'dependencies': ['overleaf'],
            'location': 'local',
        },
    }
    
    def __init__(self, narval_job_id: Optional[str] = None, auto_push: bool = False,
                 config_file: str = 'pipeline_config.json'):
        self.narval_job_id = narval_job_id
        self.auto_push = auto_push
        self.config_file = Path(config_file)
        self.status_log = Path('pipeline_status.log')
        self.phase_results = {}
    
    def check_narval_job_status(self, job_id: str) -> Optional[str]:
        """Check SLURM job status on Narval cluster.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            Job status ('PD', 'R', 'CA', 'CD', etc.) or None
        """
        try:
            result = subprocess.run(
                [
                    'ssh', 'narval',
                    f'squeue -j {job_id} --format=%T --noheader | tr -d " "'
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            status = result.stdout.strip()
            return status if status else None
        except Exception as e:
            logger.warning(f"Could not check Narval job status: {e}")
            return None
    
    def wait_for_phase1(self, poll_interval: int = 300) -> bool:
        """Monitor Phase 1 training on Narval until completion.
        
        Args:
            poll_interval: Seconds between status checks
            
        Returns:
            True if completed successfully
        """
        logger.info("Monitoring Phase 1 training on Narval")
        
        if not self.narval_job_id:
            logger.warning("No Narval job ID provided; skipping Phase 1 wait")
            return True
        
        logger.info(f"Watching job {self.narval_job_id}")
        
        while True:
            status = self.check_narval_job_status(self.narval_job_id)
            
            if status == 'CD':  # Completed
                logger.info("✓ Phase 1 completed successfully")
                self.PHASES['phase1']['status'] = 'completed'
                return True
            elif status == 'CA':  # Cancelled
                logger.error("✗ Phase 1 job cancelled")
                self.PHASES['phase1']['status'] = 'failed'
                return False
            elif status == 'F':  # Failed
                logger.error("✗ Phase 1 job failed")
                self.PHASES['phase1']['status'] = 'failed'
                return False
            elif status in ['R', 'PD']:  # Running or Pending
                logger.info(f"Phase 1 status: {status} (check back in {poll_interval}s)")
                time.sleep(poll_interval)
            else:
                logger.info(f"Phase 1 status: {status}")
                time.sleep(poll_interval)
    
    def run_local_phase(self, phase_name: str, script: str, args: List[str]) -> bool:
        """Run a local training/analysis phase.
        
        Args:
            phase_name: Phase identifier
            script: Python script to run
            args: Command-line arguments
            
        Returns:
            True if successful
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Launching {phase_name}: {self.PHASES[phase_name]['name']}")
        logger.info(f"{'='*80}")
        
        cmd = ['python', script] + args
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, timeout=86400, check=True)  # 24h timeout
            logger.info(f"✓ {phase_name} completed")
            self.PHASES[phase_name]['status'] = 'completed'
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {phase_name} failed with exit code {e.returncode}")
            self.PHASES[phase_name]['status'] = 'failed'
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"✗ {phase_name} timed out")
            self.PHASES[phase_name]['status'] = 'failed'
            return False
    
    def launch_phase(self, phase_name: str) -> bool:
        """Launch a specific phase based on its configuration.
        
        Args:
            phase_name: Phase to launch
            
        Returns:
            Success status
        """
        phase_config = self.PHASES[phase_name]
        
        # Check dependencies
        for dep in phase_config['dependencies']:
            if self.PHASES[dep]['status'] != 'completed':
                logger.info(f"Waiting for dependency: {dep}")
                return False
        
        # Phase 1 has special handling (on Narval)
        if phase_name == 'phase1':
            return self.wait_for_phase1()
        
        # Local phases
        script = phase_config.get('script')
        if not script:
            logger.warning(f"No script defined for {phase_name}")
            return False
        
        # Prepare arguments
        args = []
        if phase_name == 'phase2':
            args = [
                '--data', 'data/processed/crispro_features.pkl',
                '--output', 'checkpoints/phase2_xgboost/',
                '--n_trials', '100',
            ]
        elif phase_name == 'phase3':
            args = [
                '--phase1_checkpoint', 'checkpoints/phase1/best_model.pt',
                '--phase2_model', 'checkpoints/phase2_xgboost/xgboost_model.pkl',
                '--data', 'data/processed/crispro_dataset.pkl',
                '--output', 'checkpoints/phase3_deephybrid/',
            ]
        elif phase_name == 'phase4':
            args = [
                '--model', 'checkpoints/phase3_deephybrid/best_model.pt',
                '--clinical_data', 'data/clinical_datasets/',
                '--output', 'checkpoints/phase4_validation/',
            ]
        elif phase_name == 'benchmarking':
            args = [
                '--model', 'checkpoints/phase3_deephybrid/best_model.pt',
                '--output', 'results/benchmarking/',
            ]
        elif phase_name == 'figures':
            args = [
                '--results_dir', 'results/benchmarking/',
                '--output', 'figures/',
            ]
        elif phase_name == 'overleaf':
            args = [
                '--results_dir', 'results/benchmarking/',
                '--figures_dir', 'figures/',
            ]
        
        return self.run_local_phase(phase_name, script, args)
    
    def commit_and_tag(self) -> bool:
        """Commit all results to git with version tags."""
        logger.info(f"\n{'='*80}")
        logger.info("Committing results to GitHub")
        logger.info(f"{'='*80}")
        
        try:
            # Add all changes
            subprocess.run(['git', 'add', '-A'], check=True)
            
            # Commit
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_message = (
                f"feat: complete ChromaGuide pipeline execution\n\n"
                f"- Phase 1: DNABERT-Mamba training\n"
                f"- Phase 2: CRISPRO-XGBoost benchmarking\n"
                f"- Phase 3: DeepHybrid ensemble\n"
                f"- Phase 4: Clinical validation\n"
                f"- SOTA benchmarking and comparison\n"
                f"- Publication figures generated\n"
                f"- Overleaf paper updated with results\n\n"
                f"Timestamp: {timestamp}"
            )
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            
            # Create version tag
            version_tag = f"v2.0-complete-{datetime.now().strftime('%Y%m%d')}"
            subprocess.run(
                ['git', 'tag', '-a', version_tag, '-m', f"Complete pipeline execution {timestamp}"],
                check=True
            )
            
            logger.info(f"✓ Committed with tag: {version_tag}")
            
            # Push if requested
            if self.auto_push:
                subprocess.run(['git', 'push', 'origin', 'main'], check=True)
                subprocess.run(['git', 'push', 'origin', '--tags'], check=True)
                logger.info("✓ Pushed to GitHub")
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Git commit failed: {e}")
            return False
    
    def run_pipeline(self) -> bool:
        """Run full pipeline in order."""
        logger.info("="*80)
        logger.info("CHROMAGUIDE AUTONOMOUS PIPELINE ORCHESTRATION")
        logger.info("="*80)
        
        phase_order = [
            'phase1', 'phase2', 'phase3', 'phase4',
            'benchmarking', 'figures', 'overleaf', 'commit'
        ]
        
        for phase_name in phase_order:
            logger.info(f"\n[{datetime.now()}] Processing {phase_name}")
            
            success = self.launch_phase(phase_name)
            
            if not success:
                logger.error(f"✗ Pipeline halted at {phase_name}")
                self.save_status()
                return False
            
            # Save status after each phase
            self.save_status()
        
        logger.info("\n" + "="*80)
        logger.info("✓ PIPELINE COMPLETE!")
        logger.info("="*80)
        
        return True
    
    def save_status(self) -> None:
        """Save current pipeline status."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'phases': self.PHASES,
        }
        with open(self.status_log, 'w') as f:
            json.dump(status, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Phase Autonomous Pipeline Orchestrator'
    )
    parser.add_argument('--watch-job', type=str, default=None,
                       help='Narval job ID to monitor (e.g., 56644478)')
    parser.add_argument('--auto-push-github', action='store_true',
                       help='Automatically push results to GitHub')
    parser.add_argument('--skip-phase1-wait', action='store_true',
                       help='Skip waiting for Phase 1 (assume complete)')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(
        narval_job_id=args.watch_job,
        auto_push=args.auto_push_github
    )
    
    # Run pipeline
    success = orchestrator.run_pipeline()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
