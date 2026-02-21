#!/bin/bash
# Monitor production jobs for ChromaGuide PhD Proposal
# Usage: bash scripts/monitor_jobs.sh

# User's cluster name (provided by user for Narval)
# Note: Using the provided persistent SSH socket at /tmp/narval_socket
ssh -S /tmp/narval_socket narval "squeue -u \$USER; sacct -j 56763338,56763339,56763340 --format=JobID,JobName%30,State,Elapsed,ExitCode"
