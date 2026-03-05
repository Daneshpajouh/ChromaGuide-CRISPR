#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <cluster-host> <job-kind> [VAR=value ...]" >&2
  echo "job-kind: on-target-optuna | on-target-full | off-target-cclmoff | off-target-optuna" >&2
  exit 1
fi

CLUSTER_HOST="$1"
JOB_KIND="$2"
shift 2

case "$JOB_KIND" in
  on-target-optuna)
    REMOTE_SCRIPT="scripts/slurm_public_on_target_optuna.sh"
    ;;
  on-target-full)
    REMOTE_SCRIPT="scripts/slurm_public_on_target_full_run.sh"
    ;;
  off-target-cclmoff)
    REMOTE_SCRIPT="scripts/slurm_public_off_target_cclmoff.sh"
    ;;
  off-target-optuna)
    REMOTE_SCRIPT="scripts/slurm_public_off_target_optuna.sh"
    ;;
  *)
    echo "Unknown job kind: $JOB_KIND" >&2
    exit 1
    ;;
esac

SSH_CMD=(ssh)
for candidate in "$HOME/.ssh/cm/${CLUSTER_HOST}-"*; do
  if [ -S "$candidate" ]; then
    SSH_CMD=(ssh -o ControlPath="$candidate")
    break
  fi
done
if [ "${SSH_CMD[0]}" = "ssh" ]; then
  for candidate in "$HOME/.ssh/sockets/${CLUSTER_HOST}-"*; do
    if [ -S "$candidate" ]; then
      SSH_CMD=(ssh -o ControlPath="$candidate")
      break
    fi
  done
fi

ENV_PREFIX=""
REMOTE_REPO_DIR=""
for kv in "$@"; do
  ENV_PREFIX+="${kv} "
  case "$kv" in
    REPO_DIR=*)
      REMOTE_REPO_DIR="${kv#REPO_DIR=}"
      ;;
  esac
done

if [ -z "$REMOTE_REPO_DIR" ]; then
  REMOTE_REPO_DIR="\$HOME/chromaguide_experiments"
fi

SBATCH_BIN="$( "${SSH_CMD[@]}" "$CLUSTER_HOST" 'if command -v sbatch >/dev/null 2>&1; then command -v sbatch; elif [ -x /opt/software/slurm/bin/sbatch ]; then echo /opt/software/slurm/bin/sbatch; else echo ""; fi' )"
if [ -z "$SBATCH_BIN" ]; then
  echo "Skipping ${CLUSTER_HOST}: sbatch not available" >&2
  exit 0
fi

REMOTE_CMD="cd ${REMOTE_REPO_DIR} && ${ENV_PREFIX}${SBATCH_BIN} ${REMOTE_SCRIPT}"
echo "Submitting to ${CLUSTER_HOST}: ${REMOTE_CMD}"
if ! "${SSH_CMD[@]}" "$CLUSTER_HOST" "test -d ${REMOTE_REPO_DIR}"; then
  echo "Skipping ${CLUSTER_HOST}: ${REMOTE_REPO_DIR} missing" >&2
  exit 0
fi
"${SSH_CMD[@]}" "$CLUSTER_HOST" "$REMOTE_CMD"
