#!/usr/bin/env bash
# Wait for the running laptop scheduler (pid $1) to finish, then start it again: the new start requeues runs
# lost to provider errors (except SKIP_RETRY) and builds that failed. Same spend cap and baseline.
#   FORENSICS_ENV_FILE=... CAP_USD=380 START_USD=... SKIP_RETRY=... nohup caffeinate -ims bash laptop_retry_after.sh <pid> &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') first pass over: starting the retry pass" >> "$HERE/results/logs/full_laptop/progress.txt"
python3 "$HERE/scripts/laptop_full.py" >> "$HERE/results/logs/full_laptop_stdout.log" 2>&1
