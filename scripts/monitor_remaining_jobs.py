#!/usr/bin/env python3
"""
Monitor jobs 56706055 and 56706056, download results when complete.
Runs continuously every 5 minutes until both jobs finish.
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

JOBS_TO_MONITOR = ["56706055", "56706056"]
SSH_CONTROL_PATH = "~/.ssh/sockets/narval"
NARVAL_HOST = "amird@narval.computecanada.ca"
NARVAL_RESULTS_DIR = "/home/amird/chromaguide_experiments/results/"
LOCAL_RESULTS_DIR = Path("results/narval")
POLL_INTERVAL = 300  # 5 minutes

LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def check_job_status(job_id):
    """Check status of a single job"""
    try:
        cmd = [
            "ssh", "-o", f"ControlPath={SSH_CONTROL_PATH}", NARVAL_HOST,
            f"sacct -j {job_id} --format=State,ExitCode,Elapsed --noheader --parsable2"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None, None, None

        lines = result.stdout.strip().split('\n')
        if lines:
            state, exit_code, elapsed = lines[0].split('|')
            return state.strip(), exit_code.strip(), elapsed.strip()
    except Exception as e:
        print(f"  Error checking job {job_id}: {e}")
    return None, None, None

def download_job_results(job_id):
    """Download results from a completed job"""
    job_names = {
        "56706055": "seq_only_baseline",
        "56706056": "chromaguide_full"
    }

    job_name = job_names.get(job_id, f"job_{job_id}")
    narval_path = f"{NARVAL_RESULTS_DIR}{job_name}/"
    local_path = LOCAL_RESULTS_DIR / job_name

    print(f"  📥 Downloading {job_name} from Narval...")
    try:
        cmd = [
            "scp", "-r", "-o", f"ControlPath={SSH_CONTROL_PATH}",
            f"{NARVAL_HOST}:{narval_path}", str(local_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"  ✓ Downloaded {job_name} successfully")
            return True
        else:
            print(f"  ✗ Download failed for {job_name}")
            return False
    except Exception as e:
        print(f"  Error downloading {job_name}: {e}")
        return False

def main():
    print(f"\n{'='*80}")
    print("MONITORING JOBS 56706055 & 56706056")
    print(f"{'='*80}\n")

    completed_jobs = set()
    started_time = datetime.now()

    while len(completed_jobs) < len(JOBS_TO_MONITOR):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed_hours = (datetime.now() - started_time).total_seconds() / 3600

        print(f"\n[{timestamp}] Status check (elapsed: {elapsed_hours:.1f}h)")
        print("-" * 80)

        for job_id in JOBS_TO_MONITOR:
            if job_id in completed_jobs:
                continue

            state, exit_code, elapsed = check_job_status(job_id)

            if state is None:
                print(f"  Job {job_id}: Unable to get status")
                continue

            job_names = {
                "56706055": "seq_only_baseline",
                "56706056": "chromaguide_full"
            }
            job_name = job_names.get(job_id, f"job_{job_id}")

            if "COMPLETED" in state or "COMPLETED" == state:
                print(f"  ✅ Job {job_id} ({job_name}): COMPLETED ({elapsed})")

                # Try to download results
                if download_job_results(job_id):
                    completed_jobs.add(job_id)
                    print(f"     → Ready for analysis in results/narval/{job_name}/")
                else:
                    print(f"     ⚠️  Download failed, will retry")

            elif "FAILED" in state or "FAILED" == state:
                print(f"  ❌ Job {job_id} ({job_name}): FAILED (exit: {exit_code})")
                print(f"     Check logs: ssh narval 'tail -50 /home/amird/chromaguide_experiments/logs/{job_name}.err'")
                completed_jobs.add(job_id)

            elif "RUNNING" in state or "RUNNING" == state:
                print(f"  🔄 Job {job_id} ({job_name}): RUNNING ({elapsed})")

            elif "CANCELLED" in state:
                print(f"  🚫 Job {job_id} ({job_name}): CANCELLED")
                completed_jobs.add(job_id)
            else:
                print(f"  ⏳ Job {job_id} ({job_name}): {state} ({elapsed})")

        if len(completed_jobs) < len(JOBS_TO_MONITOR):
            print(f"\nWaiting {POLL_INTERVAL}s before next check...")
            time.sleep(POLL_INTERVAL)

    print(f"\n{'='*80}")
    print("ALL MONITORED JOBS COMPLETED!")
    print(f"{'='*80}")

    print("\n📊 NEXT STEPS:")
    print("  1. Run comprehensive analysis on all 6 completed jobs")
    print("     → python3 scripts/comprehensive_analysis.py")
    print("  2. Generate final publication figures")
    print("  3. Update documentation")
    print("  4. Begin real data retraining (REAL_DATA_RETRAINING_PLAN.md)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoring paused by user. Resume with: python3 scripts/monitor_remaining_jobs.py")

