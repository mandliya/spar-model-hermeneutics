#!/usr/bin/env bash
# Entry point for every pipeline step.
#
#   ./run.sh extract  --dataset aisi_city_train               # GPU: replay + store hidden states
#   ./run.sh extract  --dataset aisi_city_train --limit 8     # GPU smoke test
#   ./run.sh evaluate --dataset aisi_city_train               # CPU: layer fingerprint
#   ./run.sh evaluate --dataset aisi_varied_deception
#   ./run.sh extract  --dataset ascii_sandbagging --followup all
#   ./run.sh extract  --dataset oa_mbpp --followup dyl dyl_truthful --limit 8
#   ./run.sh compare_wordings                                 # CPU: just asking vs the probe, best wording each
#
# Every invocation writes its own raw log to scratch/logs/run_<utc>_<command>.log:
# the exact argv, the environment (python, nvidia-smi, pip freeze), the complete
# output and the exit code. The log is pushed to the Hub when the step ends,
# success or failure, so a crash's traceback is kept too. Long steps also push it
# with every row sync.
#
# No venv on the GPU box: it exists for one job, so deps go straight into the
# image's python with `pip install -r requirements.txt`; torch comes from the image.
#
# Safe to kill and rerun. extract skips rows already stored locally or on the Hub.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

[[ -f .env ]] || { echo "No .env here - copy .env.example and fill it in." >&2; exit 1; }
# tr strips Windows line endings, which otherwise end up inside the token.
set -a
# shellcheck disable=SC1091
source <(tr -d '\r' < .env)
set +a

command="${1:?usage: ./run.sh <extract|evaluate|compare_wordings> [args]}"
shift
case "$command" in
  extract | evaluate | compare_wordings) ;;
  *) echo "unknown command: $command (expected extract, evaluate or compare_wordings)" >&2; exit 1 ;;
esac

if [[ -z "${TMUX:-}" ]] && command -v tmux >/dev/null 2>&1; then
  echo "Not inside tmux. A dropped SSH session kills the run (resume makes that cheap, not free)." >&2
fi

LOG_DIR="$HERE/scratch/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_$(date -u +%Y%m%dT%H%M%SZ)_${command}.log"
started=$(date +%s)
{
  printf '### %s  host=%s  cwd=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)" "$PWD"
  printf '$ ./run.sh'
  printf ' %q' "$command" "$@"
  printf '\n--- python\n'
  python --version
  printf -- '--- nvidia-smi\n'
  if command -v nvidia-smi >/dev/null; then nvidia-smi; else echo "nvidia-smi not present on this host"; fi
  printf -- '--- pip freeze\n'
  python -m pip freeze
  printf -- '--- output\n'
} >>"$LOG" 2>&1
echo "logging to $LOG"

set +e
PYTHONUNBUFFERED=1 python -m "dyl_repro.cli.$command" "$@" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
printf '### exit=%s  seconds=%s\n' "$status" "$(( $(date +%s) - started ))" | tee -a "$LOG"

python -m dyl_repro.core.logs "$LOG_DIR"
exit "$status"
