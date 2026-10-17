#!/usr/bin/env bash
# Stops this Vast instance when the work finishes, OR at a hard deadline,
# whichever comes first. It ALWAYS stops.
#
# Stopping is safe unconditionally: it halts GPU charges while preserving
# /workspace, so the model cache, every result file and every activation
# survive intact and the instance can simply be restarted. An earlier version
# of this script refused to stop when the HF upload failed, on the theory that
# stranded work is worse than extra rental. That was wrong twice over: nothing
# is stranded by a stop, and it risked leaving the box billing unattended
# overnight, which is the single failure mode this project cares most about.
cd /workspace/lies-experiment

say () { echo "[$(date -u +%H:%M:%SZ)] [autostop] $*"; }

DEADLINE=$(( $(date +%s) + 7200 ))   # hard stop 2h from launch, regardless

say "armed. hard deadline 2h from now; will stop sooner once the chain exits"

while pgrep -f '^bash run_capture_followups\.sh$' >/dev/null; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    say '!!! HARD DEADLINE reached with the chain still running - stopping anyway'
    say '!!! /workspace is preserved; restart the instance to resume'
    bash stop_self.sh
    exit 0
  fi
  sleep 30
done

say 'chain has exited; giving the upload 90s to settle'
sleep 90
grep -E 'VERIFIED|ABORT|FAILED|COMPLETE' results/capture_followups2.log 2>/dev/null | tail -5
say 'stopping the instance now'
bash stop_self.sh
