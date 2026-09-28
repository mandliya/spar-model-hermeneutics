#!/usr/bin/env bash
# Show the newest run log's output in readable form: tqdm redraws collapsed.
# Display only; the raw log on disk is never modified.
#
#   ./tail_run.sh                       # newest run log, now
#   ./tail_run.sh evaluate              # newest evaluate log
#   ./tail_run.sh --wait 540 evaluate   # block up to 540s for it to end, then show it
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

wait_seconds=0
if [[ "${1:-}" == "--wait" ]]; then
  wait_seconds="${2:?--wait needs seconds}"
  shift 2
fi
command="${1:-}"

log="$(ls -t scratch/logs/run_*"${command:+_$command}".log | head -1)"
deadline=$(( $(date +%s) + wait_seconds ))
until grep -q '^### exit=' "$log" || (( $(date +%s) >= deadline )); do sleep 10; done

echo "== $log"
sed -n '/^--- output$/,$p' "$log" | tr '\r' '\n' | grep -v '^[[:space:]]*$' | awk '!seen[$0]++' | tail -n 40
