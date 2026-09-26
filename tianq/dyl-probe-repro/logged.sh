#!/usr/bin/env bash
# Run one command and append its exact text, timestamp, full output and exit
# code to a session log. Every command on the GPU box goes through this, so the
# log holds the whole session with the commands themselves, not only their
# output, which is all `tee` alone would keep.
#
#   ./logged.sh 'pip install -r requirements.txt'
#   ./logged.sh 'tmux new -d -s extract "./run.sh extract --dataset aisi_city_train"'
#
# The command is one string, logged verbatim and run with `bash -c`.
# DYL_SESSION_LOG overrides the log path; the laptop side uses it for the
# orchestration log (vastai, scp). Logs reach the Hub via dyl_repro/logs.py.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${DYL_SESSION_LOG:-$HERE/scratch/logs/session.log}"
mkdir -p "$(dirname "$LOG")"

command="${1:?usage: ./logged.sh '<command>'}"
started=$(date +%s)
printf '\n### %s  host=%s  cwd=%s\n$ %s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)" "$PWD" "$command" | tee -a "$LOG"
bash -c "$command" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
printf '### exit=%s  seconds=%s\n' "$status" "$(( $(date +%s) - started ))" | tee -a "$LOG"
exit "$status"
