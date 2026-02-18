#!/usr/bin/env python3
"""
Real-Time Monitoring Dashboard for ChromaGuide Pipeline
========================================================

Monitors Phase 1 training on Narval cluster and displays:
- Job status (queued/running/complete)
- GPU usage
- Training progress (epochs, loss, metrics)
- Estimated time to completion
- Alert notifications

Usage:
    python monitor_pipeline.py --job 56644478 --refresh 30
    
    Or use in terminal:
    watch -n 30 'python monitor_pipeline.py --job 56644478'
"""

import argparse
import logging
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineMonitor:
    """Real-time monitoring of ChromaGuide pipeline."""
    
    def __init__(self, job_id: str, host: str = 'narval'):
        self.job_id = job_id
        self.host = host
        self.project_path = '~/crispro_project'
    
    def get_job_status(self) -> Optional[Dict]:
        """Get SLURM job status.
        
        Returns:
            Dictionary with job information
        """
        try:
            cmd = [
                'ssh', self.host,
                f'squeue -j {self.job_id} --format="%i %T %e %C %m %R %S %L" --noheader'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if not result.stdout.strip():
                return None
            
            parts = result.stdout.strip().split()
            if len(parts) < 4:
                return None
            
            return {
                'job_id': parts[0],
                'status': parts[1],  # PD, R, CA, CD, F, etc.
                'exit_code': parts[2],
                'cpus': parts[3] if len(parts) > 3 else 'N/A',
                'memory': parts[4] if len(parts) > 4 else 'N/A',
                'reason': parts[5] if len(parts) > 5 else 'N/A',
                'start_time': parts[6] if len(parts) > 6 else 'N/A',
                'time_left': parts[7] if len(parts) > 7 else 'N/A',
            }
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return None
    
    def get_gpu_usage(self) -> Optional[Dict]:
        """Get GPU usage on Narval.
        
        Returns:
            Dictionary with GPU info
        """
        try:
            cmd = [
                'ssh', self.host,
                'nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total '
                '--format=csv,noheader,nounits | head -1'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if not result.stdout.strip():
                return None
            
            parts = result.stdout.strip().split(',')
            
            return {
                'gpu_util': f"{parts[1].strip()}%",
                'mem_util': f"{parts[2].strip()}%",
                'mem_used': f"{parts[3].strip()}MB",
                'mem_total': f"{parts[4].strip()}MB",
            }
        except Exception as e:
            logger.debug(f"Could not get GPU usage: {e}")
            return None
    
    def get_training_progress(self) -> Optional[Dict]:
        """Extract training progress from logs.
        
        Returns:
            Dictionary with training metrics
        """
        try:
            cmd = [
                'ssh', self.host,
                f'tail -50 {self.project_path}/logs/phase1_{self.job_id}.out'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            output = result.stdout
            
            # Parse for epoch and loss information
            epoch_match = re.search(r'Epoch (\d+)/\d+', output)
            loss_match = re.search(r'Loss: ([\d.]+)', output)
            spearman_match = re.search(r'Spearman r: ([\d.]+)', output)
            
            progress = {}
            
            if epoch_match:
                progress['current_epoch'] = int(epoch_match.group(1))
            
            if loss_match:
                progress['loss'] = float(loss_match.group(1))
            
            if spearman_match:
                progress['spearman_r'] = float(spearman_match.group(1))
            
            # Try to extract training history
            if progress:
                return progress
            
            # Check for job just started
            if 'Starting training' in output or 'Epoch 1' in output:
                return {'status': 'training_started'}
            
            return None
        except Exception as e:
            logger.debug(f"Could not get training progress: {e}")
            return None
    
    def estimate_time_to_completion(self, job_info: Dict, progress: Optional[Dict]) -> Optional[str]:
        """Estimate time to completion.
        
        Args:
            job_info: Job status dictionary
            progress: Training progress dictionary
            
        Returns:
            Estimated completion time or None
        """
        try:
            # Parse time_left from job info
            time_left_str = job_info.get('time_left', '')
            
            # Format: "1-05:30:00" (days-hours:minutes:seconds)
            if '-' in time_left_str:
                days, time_part = time_left_str.split('-')
                hours, minutes, seconds = time_part.split(':')
                remaining_seconds = (int(days) * 86400 + int(hours) * 3600 + 
                                   int(minutes) * 60 + int(seconds))
            else:
                # Format: "05:30:00"
                try:
                    hours, minutes, seconds = time_left_str.split(':')
                    remaining_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
                except:
                    remaining_seconds = None
            
            if remaining_seconds and remaining_seconds > 0:
                completion_time = datetime.now() + timedelta(seconds=remaining_seconds)
                return completion_time.strftime("%Y-%m-%d %H:%M:%S")
            
            return None
        except Exception as e:
            logger.debug(f"Could not estimate completion time: {e}")
            return None
    
    def display_dashboard(self) -> None:
        """Display real-time monitoring dashboard."""
        # Get all info
        job_info = self.get_job_status()
        gpu_info = self.get_gpu_usage()
        progress = self.get_training_progress()
        
        # Clear screen and display
        print("\033[2J\033[H")  # Clear screen
        
        print("="*80)
        print("CHROMAGUIDE PHASE 1 PIPELINE MONITOR")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Job status
        if job_info:
            print("\n[JOB STATUS]")
            print(f"  Job ID:     {job_info['job_id']}")
            print(f"  Status:     {job_info['status']}")
            
            status_colors = {
                'PD': '🟡',  # Pending
                'R': '🟢',   # Running
                'CA': '🔴',  # Cancelled
                'CD': '✅',  # Completed
                'F': '❌',   # Failed
            }
            
            status_emoji = status_colors.get(job_info['status'], '❓')
            print(f"  Status Emoji: {status_emoji}")
            
            if job_info['status'] == 'R':
                print(f"  CPUs:       {job_info['cpus']}")
                print(f"  Memory:     {job_info['memory']}")
                
        # GPU usage
        if gpu_info:
            print("\n[GPU USAGE]")
            print(f"  GPU Utilization: {gpu_info['gpu_util']}")
            print(f"  Memory Used:     {gpu_info['mem_used']} / {gpu_info['mem_total']}")
            print(f"  Memory Util:     {gpu_info['mem_util']}")
        
        # Training progress
        if progress:
            print("\n[TRAINING PROGRESS]")
            if 'current_epoch' in progress:
                print(f"  Epoch:       {progress['current_epoch']}/10")
            if 'loss' in progress:
                print(f"  Loss:        {progress['loss']:.6f}")
            if 'spearman_r' in progress:
                print(f"  Spearman r:  {progress['spearman_r']:.4f}")
            if 'status' in progress:
                print(f"  Status:      {progress['status']}")
        
        # Time to completion
        if job_info and progress:
            eta = self.estimate_time_to_completion(job_info, progress)
            if eta:
                print(f"\n[ESTIMATED COMPLETION]")
                print(f"  Time: {eta}")
        
        print("\n" + "="*80)
        
        # Next actions
        if job_info:
            if job_info['status'] == 'PD':
                print("Status: QUEUED - Waiting for GPU allocation...")
            elif job_info['status'] == 'R':
                print("Status: TRAINING - GPU training in progress")
            elif job_info['status'] == 'CD':
                print("Status: COMPLETE - Phase 1 training finished!")
                print("Next steps:")
                print("  1. Download results: scp -r narval:~/crispro_project/checkpoints/phase1/ .")
                print("  2. Start Phase 2: python train_phase2_xgboost.py ...")
            elif job_info['status'] == 'F':
                print("Status: FAILED - Check error logs")
                print(f"  Command: ssh {self.host} tail ~/crispro_project/logs/phase1_{self.job_id}.err")
        
        print("="*80)
    
    def run_continuous_monitoring(self, refresh_interval: int = 30) -> None:
        """Run continuous monitoring loop.
        
        Args:
            refresh_interval: Seconds between refreshes
        """
        import time
        
        logger.info(f"Starting continuous monitoring (refresh every {refresh_interval}s)")
        
        try:
            while True:
                self.display_dashboard()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")


def main():
    parser = argparse.ArgumentParser(description='Pipeline monitoring dashboard')
    parser.add_argument('--job', type=str, default='56644478',
                       help='SLURM job ID to monitor')
    parser.add_argument('--host', type=str, default='narval',
                       help='Cluster hostname')
    parser.add_argument('--refresh', type=int, default=30,
                       help='Refresh interval in seconds')
    parser.add_argument('--once', action='store_true',
                       help='Show status once and exit (default: continuous)')
    
    args = parser.parse_args()
    
    monitor = PipelineMonitor(job_id=args.job, host=args.host)
    
    if args.once:
        monitor.display_dashboard()
    else:
        monitor.run_continuous_monitoring(refresh_interval=args.refresh)


if __name__ == '__main__':
    main()
