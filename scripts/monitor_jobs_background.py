#!/usr/bin/env python3
"""
Background monitoring loop for ChromaGuide training jobs
Checks job status every 10 minutes and downloads results as they complete
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import os

RESULTS_DIR = Path("/Users/studio/Desktop/PhD/Proposal/results/completed_jobs")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Jobs to monitor
JOBS = {
    '56685445': 'seq_only_baseline',
    '56685446': 'chromaguide_full',
    '56685447': 'mamba_variant',
    '56685448': 'ablation_fusion',
    '56685449': 'ablation_modality',
    '56685450': 'hpo_optuna'
}

# Track downloaded jobs
downloaded = set()
check_interval = 600  # 10 minutes in seconds

def log_message(msg):
    """Print timestamped message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

def get_job_status():
    """Check status of all jobs"""
    try:
        result = subprocess.run(
            ['ssh', 'narval', 'squeue -u amird --format="%.7i %.15j %.2t %.5M" 2>&1'],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout
    except Exception as e:
        log_message(f"Error checking job status: {e}")
        return None

def download_results(job_id, job_name):
    """Download results from completed job"""
    try:
        log_message(f"Downloading results for job {job_id} ({job_name})")
        
        # Try to download results.json
        src = f"narval:/home/amird/chromaguide_experiments/results/{job_name}/results.json"
        dest = RESULTS_DIR / f"{job_name}_results.json"
        
        result = subprocess.run(
            ['scp', src, str(dest)],
            capture_output=True,
            timeout=60
        )
        
        if result.returncode == 0:
            log_message(f"  ✓ Downloaded results.json")
            return True
        else:
            # Try comparison.json for ablation jobs
            src = f"narval:/home/amird/chromaguide_experiments/results/{job_name}/comparison.json"
            result = subprocess.run(
                ['scp', src, str(dest)],
                capture_output=True,
                timeout=60
            )
            if result.returncode == 0:
                log_message(f"  ✓ Downloaded comparison.json")
                return True
            else:
                log_message(f"  ✗ Could not download results")
                return False
                
    except Exception as e:
        log_message(f"Error downloading results: {e}")
        return False

def check_job_completion(job_id):
    """Check if specific job is completed"""
    try:
        result = subprocess.run(
            ['ssh', 'narval', f'sacct -j {job_id} --format=state --noheader 2>&1'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout.strip().split('\n')[0].strip()
        
        # Look for COMPLETED, FAILED, or other end states
        if 'COMPLETED' in output:
            return 'COMPLETED'
        elif 'FAILED' in output:
            return 'FAILED'
        elif 'RUNNING' in output or 'PENDING' in output:
            return 'ACTIVE'
        else:
            return output
            
    except Exception as e:
        log_message(f"Error checking job {job_id}: {e}")
        return None

def monitoring_loop():
    """Main monitoring loop"""
    log_message("=" * 80)
    log_message("STARTING CHROMAGUIDE JOB MONITORING")
    log_message("=" * 80)
    log_message(f"Monitoring {len(JOBS)} jobs every {check_interval} seconds")
    log_message(f"Results directory: {RESULTS_DIR}")
    log_message("")
    
    iteration = 0
    completed_jobs = {}
    
    while True:
        iteration += 1
        log_message(f"--- CHECK #{iteration} ---")
        
        # Check each job
        for job_id, job_name in JOBS.items():
            status = check_job_completion(job_id)
            
            if status == 'COMPLETED' and job_id not in downloaded:
                log_message(f"Job {job_id} ({job_name}): COMPLETED ✓")
                
                # Try to download results
                if download_results(job_id, job_name):
                    downloaded.add(job_id)
                    completed_jobs[job_id] = {
                        'name': job_name,
                        'status': 'RESULTS_DOWNLOADED',
                        'time': datetime.now().isoformat()
                    }
                else:
                    log_message(f"  Will retry downloading results next iteration")
                    
            elif status == 'COMPLETED':
                log_message(f"Job {job_id} ({job_name}): COMPLETED (results already downloaded)")
                
            elif status == 'FAILED':
                log_message(f"Job {job_id} ({job_name}): FAILED ✗")
                
            elif status == 'ACTIVE':
                # Get current runtime
                try:
                    result = subprocess.run(
                        ['ssh', 'narval', f'squeue -j {job_id} --format="%.5M" --noheader 2>&1'],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    runtime = result.stdout.strip() if result.stdout else "?"
                    log_message(f"Job {job_id} ({job_name}): RUNNING ({runtime})")
                except:
                    log_message(f"Job {job_id} ({job_name}): RUNNING")
            else:
                log_message(f"Job {job_id} ({job_name}): {status}")
        
        # Check if all jobs completed
        if len(downloaded) == len(JOBS):
            log_message("")
            log_message("=" * 80)
            log_message("ALL JOBS COMPLETED AND RESULTS DOWNLOADED!")
            log_message("=" * 80)
            
            # Save completion summary
            summary = {
                'completion_time': datetime.now().isoformat(),
                'total_jobs': len(JOBS),
                'completed_jobs': completed_jobs,
                'downloaded_files': list(RESULTS_DIR.glob("*.json"))
            }
            
            with open(RESULTS_DIR / "monitoring_summary.json", 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            log_message(f"✓ Summary saved to monitoring_summary.json")
            log_message("\nAll jobs completed. Exiting monitoring loop.")
            break
        
        log_message(f"Completed: {len(downloaded)}/{len(JOBS)} jobs")
        log_message(f"Next check in {check_interval}s...")
        log_message("")
        
        # Wait before next check
        time.sleep(check_interval)

if __name__ == "__main__":
    try:
        monitoring_loop()
    except KeyboardInterrupt:
        log_message("\nMonitoring stopped by user")
    except Exception as e:
        log_message(f"Error: {e}")
