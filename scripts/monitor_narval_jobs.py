#!/usr/bin/env python3
"""
Real-time monitoring of Narval training jobs.

Monitors:
- Job status (PENDING, RUNNING, COMPLETED, FAILED)
- Resource utilization (GPU, memory, CPU)
- Training logs and intermediate outputs
- Data download progress
- Intermediate results (models, predictions)
"""

import subprocess
import json
import os
from datetime import datetime
from pathlib import Path
import time
import sys
from typing import Dict, List, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class NarvalJobMonitor:
    """Monitor jobs running on Narval cluster."""
    
    def __init__(self, user: str = "amird", host: str = "narval.alliancecan.ca"):
        self.user = user
        self.host = host
        self.ssh_cmd_prefix = f"ssh {user}@{host}"
        self.jobs = {
            56676525: {'name': 'seq_only_baseline', 'expected_time': 6},
            56676526: {'name': 'chromaguide_full', 'expected_time': 8},
            56676527: {'name': 'mamba_variant', 'expected_time': 8},
            56676528: {'name': 'ablation_fusion', 'expected_time': 8},
            56676529: {'name': 'ablation_modality', 'expected_time': 8},
            56676530: {'name': 'hpo_optuna', 'expected_time': 12},
        }
        self.status_file = Path("narval_monitoring.log")
        
    def run_ssh_command(self, cmd: str) -> str:
        """Execute command on Narval via SSH."""
        try:
            result = subprocess.run(
                f"{self.ssh_cmd_prefix} \"{cmd}\"",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error(f"SSH command timed out: {cmd}")
            return ""
        except Exception as e:
            logger.error(f"SSH error: {e}")
            return ""
    
    def get_job_status(self, job_id: int) -> Dict:
        """Get status of a specific job."""
        cmd = f"squeue -j {job_id} -o '%T %M %D %e'"
        output = self.run_ssh_command(cmd)
        
        if not output:
            return {'status': 'UNKNOWN', 'time': 'N/A', 'nodes': 'N/A', 'error': 'Unknown'}
        
        lines = output.strip().split('\n')
        if len(lines) > 1:
            parts = lines[1].split()
            if len(parts) >= 3:
                return {
                    'status': parts[0],
                    'time': parts[1],
                    'nodes': parts[2],
                    'error': parts[3] if len(parts) > 3 else 'None'
                }
        
        return {'status': 'NOT_FOUND', 'time': 'N/A', 'nodes': 'N/A', 'error': 'Job not in queue'}
    
    def get_all_jobs_status(self) -> Dict:
        """Get status of all jobs."""
        cmd = f"squeue -u {self.user} --format='%i %T %M %D'"
        output = self.run_ssh_command(cmd)
        
        jobs_status = {}
        lines = output.strip().split('\n')
        
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 4:
                job_id = int(parts[0])
                if job_id in self.jobs:
                    jobs_status[job_id] = {
                        'status': parts[1],
                        'time_elapsed': parts[2],
                        'nodes': parts[3],
                    }
        
        return jobs_status
    
    def get_job_logs(self, job_id: int) -> str:
        """Retrieve job output logs."""
        job_name = self.jobs.get(job_id, {}).get('name', f'job_{job_id}')
        cmd = f"cat /home/{self.user}/chromaguide_experiments/slurm_logs/{job_id}*.log 2>/dev/null || echo 'No logs yet'"
        return self.run_ssh_command(cmd)
    
    def get_gpu_status(self) -> str:
        """Check GPU utilization on allocated nodes."""
        cmd = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,nounits"
        return self.run_ssh_command(cmd)
    
    def get_data_status(self) -> Dict:
        """Check data download progress."""
        cmd = f"du -sh /home/{self.user}/chromaguide_data/* 2>/dev/null | tail -10"
        output = self.run_ssh_command(cmd)
        
        return {
            'output': output,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_results_status(self) -> Dict:
        """Check available results."""
        cmd = f"find /home/{self.user}/chromaguide_results -type f -mmin -60 | wc -l && du -sh /home/{self.user}/chromaguide_results 2>/dev/null"
        output = self.run_ssh_command(cmd)
        
        return {
            'output': output,
            'timestamp': datetime.now().isoformat()
        }
    
    def download_intermediate_results(self, job_id: int, 
                                     local_path: Path = Path("results")) -> bool:
        """Download intermediate results for a job."""
        job_name = self.jobs.get(job_id, {}).get('name', f'job_{job_id}')
        local_dir = local_path / job_name
        local_dir.mkdir(parents=True, exist_ok=True)
        
        remote_path = f"/home/{self.user}/chromaguide_results/models/{job_id}_*.pt"
        
        try:
            cmd = f"scp -r {self.user}@{self.host}:{remote_path} {local_dir}/ 2>/dev/null || true"
            subprocess.run(cmd, shell=True, timeout=60)
            logger.info(f"Downloaded results for job {job_id} to {local_dir}")
            return True
        except Exception as e:
            logger.warning(f"Could not download results for job {job_id}: {e}")
            return False
    
    def generate_report(self) -> str:
        """Generate comprehensive monitoring report."""
        report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║            NARVAL TRAINING JOBS - MONITORING REPORT                        ║
║            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                           ║
╚════════════════════════════════════════════════════════════════════════════╝

JOB STATUS
──────────
"""
        
        all_status = self.get_all_jobs_status()
        
        for job_id, job_info in self.jobs.items():
            status = all_status.get(job_id, {})
            job_status = status.get('status', 'NOT_FOUND')
            time_elapsed = status.get('time_elapsed', 'N/A')
            
            # Status indicator
            if job_status == 'RUNNING':
                status_icon = '🟢 RUNNING'
            elif job_status == 'PENDING':
                status_icon = '🟡 PENDING'
            elif job_status == 'COMPLETED':
                status_icon = '✅ COMPLETED'
            elif job_status == 'FAILED':
                status_icon = '❌ FAILED'
            else:
                status_icon = '❓ UNKNOWN'
            
            report += f"\n{status_icon}  Job {job_id}: {job_info['name']}\n"
            report += f"  └─ Time: {time_elapsed} / {job_info['expected_time']}h\n"
        
        # Data status
        report += "\n\nDATA STATUS\n───────────\n"
        data_status = self.get_data_status()
        report += data_status['output']
        
        # Results status
        report += "\n\nRESULTS STATUS\n──────────────\n"
        results_status = self.get_results_status()
        report += results_status['output']
        
        # Summary
        report += "\n\nSUMMARY\n───────\n"
        running_count = sum(1 for s in all_status.values() if s.get('status') == 'RUNNING')
        pending_count = sum(1 for s in all_status.values() if s.get('status') == 'PENDING')
        completed_count = sum(1 for s in all_status.values() if s.get('status') == 'COMPLETED')
        
        report += f"Running:   {running_count}/6\n"
        report += f"Pending:   {pending_count}/6\n"
        report += f"Completed: {completed_count}/6\n"
        
        total_jobs = len(all_status)
        if total_jobs > 0:
            progress = (completed_count / 6) * 100
            report += f"\nProgress: {progress:.1f}%\n"
        
        return report
    
    def save_report(self, report: str):
        """Save report to file."""
        with open(self.status_file, 'a') as f:
            f.write(report + '\n\n')
            f.write('─' * 80 + '\n\n')
    
    def monitor_loop(self, interval: int = 300, max_iterations: int = None):
        """
        Continuously monitor jobs.
        
        Args:
            interval: Check interval in seconds (default 5 minutes)
            max_iterations: Maximum number of iterations (None = infinite)
        """
        iteration = 0
        
        while max_iterations is None or iteration < max_iterations:
            try:
                report = self.generate_report()
                print(report)
                self.save_report(report)
                
                # Download intermediate results
                all_status = self.get_all_jobs_status()
                for job_id in self.jobs.keys():
                    if all_status.get(job_id, {}).get('status') == 'RUNNING':
                        self.download_intermediate_results(job_id)
                
                iteration += 1
                if max_iterations is None or iteration < max_iterations:
                    logger.info(f"Next check in {interval}s...")
                    time.sleep(interval)
                    
            except KeyboardInterrupt:
                logger.info("Monitoring interrupted by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)
        
        logger.info("Monitoring complete")


def main():
    """Main monitoring script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Narval training jobs')
    parser.add_argument('--interval', type=int, default=300, 
                       help='Check interval in seconds (default 300)')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Number of iterations (default: infinite)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit')
    parser.add_argument('--download', action='store_true',
                       help='Download results for all jobs and exit')
    
    args = parser.parse_args()
    
    monitor = NarvalJobMonitor()
    
    if args.once:
        report = monitor.generate_report()
        print(report)
        monitor.save_report(report)
        
    elif args.download:
        for job_id in monitor.jobs.keys():
            monitor.download_intermediate_results(job_id)
            
    else:
        max_iter = args.iterations if args.iterations else None
        monitor.monitor_loop(interval=args.interval, max_iterations=max_iter)


if __name__ == '__main__':
    main()
