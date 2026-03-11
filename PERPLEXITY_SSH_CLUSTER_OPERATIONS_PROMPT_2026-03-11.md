# Perplexity Prompt: SSH Persistence, Cluster Operations, Sync, Submission, Harvest, and Safety

Use this prompt exactly as written.

## Role
You are taking over the SSH, cluster-operations, sync, and job-submission side of a live PhD-thesis CRISPR benchmarking project.

This is not a generic Linux tutorial task.
This is a live operations task.
You must give exact, executable, thesis-grade operational guidance that matches the current repo and workflow state.

## Primary Context You Must Read First
You must read the following files before answering:

- `HANDOVER_FULL_CONTEXT_2026-03-11.md`
- `PUBLIC_EXECUTION_STATUS_2026-03-05.json`
- `CLUSTER_BENCHMARK_EXECUTION_STATUS_2026-03-05.json`
- `SOTA_UPSTREAM_REPRO_STATUS_2026-03-05.json`

If you cannot read those files directly, stop and ask for them. Do not continue with partial context.

## Main Goal
I need exact instructions for how to:

- create persistent SSH connections to the clusters
- require MFA approval only once at the start if needed
- keep those SSH connections alive for the full work session
- verify the connections are still alive later without redoing MFA
- test remote repo paths and virtual environments
- sync scripts and files to the remote scratch repo safely
- submit jobs correctly
- monitor jobs correctly
- harvest results correctly
- update logs and scoreboards correctly after jobs finish

## Important Real-World Constraint
If MFA is required, I am willing to approve it **once** at the beginning.
After that, the setup must maintain persistent SSH sessions so the connections remain usable without repeated MFA prompts during the session.

You must explain exactly how to do that.

## Current Cluster/Repo Assumptions You Must Respect
Unless the handover proves otherwise, assume the following are current:

- main local repo: `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR`
- active cluster work is mainly on `rorqual`
- other clusters may include `nibi`, `fir`, `cedar`, `trillium`
- remote scratch repo path on rorqual: `/scratch/amird/chromaguide_experiments`
- generic env: `/scratch/amird/env_public_benchmark`
- HNN env: `/scratch/amird/env_public_benchmark_hnn`

You must verify these assumptions against the handover if possible.

## What I Need From You
I want a complete SSH + cluster operations playbook for this project.

That includes:

### A. Persistent SSH connection setup
Give exact commands to:
- create a local `~/.ssh/config` setup or equivalent command-line setup for persistent ControlMaster sockets
- set `ControlMaster auto`
- set `ControlPersist`
- set a safe `ControlPath`
- set keepalive settings so connections stay alive
- establish the first connection to each cluster (`rorqual`, `nibi`, `fir`, and any others still relevant)
- explain how I approve MFA once and then keep the control socket alive
- explain how to test that the persistent socket is working

I want both:
1. the recommended `~/.ssh/config` block approach
2. the direct one-line `ssh -M -S ...` fallback approach

### B. Cluster session verification
Give exact commands to verify:
- the SSH control socket is live
- the remote host responds
- the remote repo path exists
- the target venv exists
- the scratch filesystem is accessible
- `sbatch`, `squeue`, and `scancel` are available

### C. Remote environment verification
I want exact commands to check:
- Python version in each venv
- whether critical packages are installed
- whether the HNN env is the right one for HNN jobs
- whether the generic env is safe for non-HNN jobs
- whether the remote path has the latest scripts

### D. Sync workflow
Give exact commands to:
- sync updated scripts to the cluster repo
- use `tar | ssh` sync safely when `rsync` is unavailable
- sync only the minimal needed files for a run
- verify the sync succeeded

### E. Job submission workflow
Give exact commands to:
- submit a job manually with `sbatch`
- submit using the helper submission script if appropriate
- inject env vars safely
- avoid the known comma-valued env truncation problems
- ensure HNN jobs pin the dedicated HNN env
- ensure run tags are explicit and traceable

### F. Monitoring workflow
Give exact commands to:
- watch queue state
- inspect a single job
- inspect logs
- inspect output directories
- tell whether a job is actually running or silently failed
- tell whether a pending job is blocked by partition / time-limit / node constraints

### G. Harvest workflow
Give exact commands to:
- copy summary JSONs back locally
- create dated harvest directories safely
- verify that the harvested result is complete
- inspect the summary quickly before updating ledgers

### H. After-harvest update workflow
Give exact commands and sequence for:
- updating status JSONs
- regenerating scoreboards if necessary
- committing changes
- pushing to GitHub

### I. Safety / failure handling
Give exact commands and rules for:
- restarting dead SSH control sockets
- refreshing a persistent session without losing work
- cleaning stale control sockets
- handling MFA-expired sessions
- switching from one cluster to another if one cluster becomes unhealthy
- testing before launching a large wave

## Required Output Structure
Return the answer in this exact structure.

### 1. Exact SSH Persistence Setup
Include both `~/.ssh/config` and direct-command approaches.

### 2. Exact First-Time MFA Workflow
Explain step-by-step what happens when I approve MFA once and how the session stays alive.

### 3. Exact Verification Commands
A clean checklist of commands and what success looks like.

### 4. Exact Sync Commands
Minimal and safe.

### 5. Exact Submission Commands
For HNN, FMC, and generic job submission patterns.

### 6. Exact Monitoring Commands
For queue, logs, and job state.

### 7. Exact Harvest Commands
For pulling results and storing them in the right place.

### 8. Exact Post-Harvest Update Procedure
Including Git discipline.

### 9. Exact Failure-Recovery Procedure
What to do if the connection dies, the MFA expires, the env is wrong, or the job fails.

### 10. Exact “Start Of Session” Checklist
What I should do at the beginning of every cluster session.

### 11. Exact “Before Large Submission Wave” Checklist
What I must verify before launching many jobs.

### 12. Exact “After Every Completed Job” Checklist
What must be done to keep the repo and thesis record correct.

## Quality Bar
Do not answer vaguely.
Do not just describe SSH concepts.
Do not say “use ControlMaster” without showing the exact config and commands.
Do not assume I will infer the details.

I want copy-pasteable commands and an operationally correct workflow.

## Final Constraint
This repo is part of my PhD thesis record.
So your answer must be:
- exact
- operational
- safe
- reproducible
- logging-aware
- Git-aware
- cluster-aware
- MFA-aware
