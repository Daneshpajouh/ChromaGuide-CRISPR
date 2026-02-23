#!/usr/bin/env python3
"""
V10 DEPLOYMENT ORCHESTRATOR

Parallel execution on:
1. Fir cluster (primary) - SLURM job submission
2. Local Mac Studio (fallback/secondary) - direct execution

Features:
- Auto-detect cluster availability
- Submit SLURM jobs to Fir if accessible
- Launch local training in parallel as fallback
- Real-time monitoring
- Git-based result synchronization
"""

import subprocess
import sys
import time
import socket
import os
from pathlib import Path
from datetime import datetime

# Colors for output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}\n")

def print_status(status, message):
    symbols = {
        'check': '✅',
        'cross': '❌',
        'arrow': '➜',
        'hourglass': '⏳',
        'rocket': '🚀',
        'globe': '🌐'
    }
    colors = {
        'success': Colors.GREEN,
        'error': Colors.RED,
        'info': Colors.BLUE,
        'warning': Colors.YELLOW,
        'pending': Colors.CYAN
    }

    symbol = list(symbols.values())[0]
    color = colors.get('info', Colors.BLUE)

    for key, sym in symbols.items():
        if key in status.lower():
            symbol = sym
    for key, col in colors.items():
        if key in status.lower():
            color = col

    print(f"{color}{symbol} {message}{Colors.END}")

def check_cluster_connectivity(hostname='fir.alliancecan.ca', timeout=5):
    """Check if Fir cluster is accessible"""
    try:
        result = subprocess.run(
            ['ssh', '-o', f'ConnectTimeout={timeout}', hostname, 'echo OK'],
            capture_output=True,
            timeout=timeout+2
        )
        return result.returncode == 0
    except:
        return False

def submit_slurm_job(script_path, cluster='fir'):
    """Submit SLURM job to cluster"""
    try:
        host = f'{cluster}.alliancecan.ca'

        # Transfer script if on local machine
        if os.uname().sysname == 'Darwin':  # macOS
            subprocess.run([
                'scp', script_path,
                f'studio@{host}:~/chromaguide/'
            ], check=True, capture_output=True)

        # Submit job
        script_name = Path(script_path).name
        result = subprocess.run([
            'ssh', f'studio@{host}',
            f'cd ~/chromaguide && sbatch {script_name}'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            return job_id
        else:
            raise Exception(result.stderr)

    except Exception as e:
        print_status('error', f"Failed to submit to {cluster}: {e}")
        return None

def monitor_slurm_job(job_id, cluster='fir', interval=60, max_time=3600):
    """Monitor SLURM job status"""
    host = f'{cluster}.alliancecan.ca'
    start_time = time.time()

    while time.time() - start_time < max_time:
        try:
            result = subprocess.run([
                'ssh', f'studio@{host}',
                f'squeue -j {job_id} -h -o "%T %P"'
            ], capture_output=True, text=True, timeout=10)

            if result.stdout.strip():
                status, partition = result.stdout.strip().split()
                if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
                    return status
                print_status('pending', f"Job {job_id} status: {status}")
            else:
                print_status('check', f"Job {job_id} completed or not found")
                return 'COMPLETED'

        except Exception as e:
            print_status('warning', f"Error checking job {job_id}: {e}")

        time.sleep(interval)

    return 'TIMEOUT'

def run_local_training(script_path):
    """Run training locally on Mac Studio"""
    try:
        print_status('hourglass', f"Launching local training: {Path(script_path).name}")

        result = subprocess.Popen(
            ['python3', script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        return result
    except Exception as e:
        print_status('error', f"Failed to launch local training: {e}")
        return None

def git_commit_results(message):
    """Commit V10 results to git"""
    try:
        os.chdir('/Users/studio/Desktop/PhD/Proposal')

        subprocess.run(['git', 'add', '-A'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', message], check=True, capture_output=True)
        subprocess.run(['git', 'push', 'origin', 'main'], check=True, capture_output=True)

        print_status('check', "Results committed and pushed to git")
        return True
    except Exception as e:
        print_status('warning', f"Git commit failed: {e}")
        return False

def main():
    print_header("V10 DEPLOYMENT ORCHESTRATOR")
    print(f"{Colors.CYAN}Parallel execution on Fir + Mac Studio{Colors.END}\n")

    # Prepare
    project_root = Path('/Users/studio/Desktop/PhD/Proposal')
    os.chdir(project_root)

    # Create logs directory
    (project_root / 'logs').mkdir(exist_ok=True)

    # Check cluster connectivity
    print_status('info', "Checking Fir cluster connectivity...")
    fir_available = check_cluster_connectivity()

    if fir_available:
        print_status('success', "✓ Fir cluster accessible!")
    else:
        print_status('warning', "✗ Fir cluster not accessible (using local only)")

    print()

    # Determine execution strategy
    jobs = {
        'multimodal': {
            'local_script': 'scripts/train_on_real_data_v10.py',
            'slurm_script': 'slurm_fir_v10_multimodal.sh',
            'name': 'V10 Multimodal (On-target)'
        },
        'offtarget': {
            'local_script': 'scripts/train_off_target_v10.py',
            'slurm_script': 'slurm_fir_v10_off_target.sh',
            'name': 'V10 Off-target'
        }
    }

    # Launch jobs
    slurm_jobs = {}
    local_processes = {}

    for job_key, job_config in jobs.items():
        print_header(job_config['name'])

        if fir_available:
            # Submit to Fir
            print_status('rocket', f"Submitting to Fir cluster: {job_config['slurm_script']}")
            job_id = submit_slurm_job(job_config['slurm_script'])

            if job_id:
                print_status('success', f"Submitted job {job_id}")
                slurm_jobs[job_key] = job_id
            else:
                print_status('warning', "Fir submission failed, falling back to local")
                proc = run_local_training(job_config['local_script'])
                if proc:
                    local_processes[job_key] = proc
        else:
            # Run locally
            proc = run_local_training(job_config['local_script'])
            if proc:
                local_processes[job_key] = proc

    print()
    print_header("EXECUTION STATUS")

    # Monitor execution
    execution_summary = {}

    # Monitor SLURM jobs
    for job_key, job_id in slurm_jobs.items():
        print_status('hourglass', f"Monitoring Fir job {job_id}: {jobs[job_key]['name']}")
        status = monitor_slurm_job(job_id)
        execution_summary[job_key] = {
            'platform': 'Fir',
            'status': status,
            'job_id': job_id
        }

    # Monitor local processes
    for job_key, proc in local_processes.items():
        print_status('hourglass', f"Monitoring local process: {jobs[job_key]['name']}")
        try:
            proc.wait(timeout=3600*8)  # 8 hour timeout for local
            status = 'COMPLETED' if proc.returncode == 0 else 'FAILED'
        except subprocess.TimeoutExpired:
            status = 'TIMEOUT'
            proc.kill()

        execution_summary[job_key] = {
            'platform': 'Mac Studio',
            'status': status,
            'pid': proc.pid
        }

    # Print summary
    print()
    print_header("EXECUTION SUMMARY")

    all_success = True
    for job_key, summary in execution_summary.items():
        status = summary['status']
        platform = summary['platform']
        symbol = '✅' if status == 'COMPLETED' else '⚠️'

        print(f"{symbol} {jobs[job_key]['name']}")
        print(f"   Platform: {platform}")
        print(f"   Status: {status}")
        if 'job_id' in summary:
            print(f"   Job ID: {summary['job_id']}")
        print()

        if status != 'COMPLETED':
            all_success = False

    # Git commit
    if all_success:
        print_status('success', "All V10 training completed successfully!")
        git_commit_results("V10 training complete: hybrid DNABERT-2 architectures with epigenetic gating")
    else:
        print_status('warning', "Some V10 training jobs did not complete")
        git_commit_results("V10 training in progress: monitoring execution")

    print()
    print_header("NEXT STEPS")
    print("""
1. Monitor training progress:
   - Fir: tail -f logs/slurm_*_v10_*.log
   - Local: tail -f logs/multimodal_v10_*.log, logs/off_target_v10_*.log

2. After training completes:
   - Run evaluation: python3 scripts/evaluate_v10_models.py
   - Compare vs targets: Multimodal Rho vs 0.911, Off-target AUROC vs 0.99

3. If targets not met:
   - Analyze gap sources
   - Plan V11 improvements (ensemble scaling, hyperparameter tuning)
   - Consider hybrid approach with other proven architectures

""")

if __name__ == '__main__':
    main()
