#!/usr/bin/env bash
# general_tbench_chain.sh (24 Nov 2026): GG's yes (24 Nov, ~$20) for the first Terminal-Bench steps, in order, on the
# launch box, stopping at the first failure: setup (free), check (free), oracle on all 89 tasks with no model (Modal,
# ~$3), smoke with DeepSeek (cap $5), pilot of 12 tasks x 2 runs (cap $60). The full collection is NOT here: it needs
# its own yes with the pilot's measured price. Each step's output: /workspace/logs/general_tbench_<step>.out; one line
# per step start and end in /workspace/logs/general_tbench.txt. Never stops a box.
#   setsid nohup bash /workspace/forensics/gigi/3-forensics/scripts/general_tbench_chain.sh > /workspace/logs/general_tbench_chain.out 2>&1 < /dev/null &
set -uo pipefail
L=/workspace/forensics/gigi/3-forensics/scripts/general_tbench_launch.sh; LOGD=/workspace/logs; S=$LOGD/general_tbench.txt
step () {   # step NAME command...
  local n=$1; shift
  echo "$(date -u +%FT%TZ) CHAIN $n: start" >> "$S"
  "$@" > "$LOGD/general_tbench_$n.out" 2>&1 < /dev/null; local rc=$?
  echo "$(date -u +%FT%TZ) CHAIN $n: exit $rc" >> "$S"; return $rc
}
step setup bash "$L" setup \
  && step check bash "$L" check \
  && step oracle env N=8 bash "$L" oracle \
  && step smoke env SPEND_CAP=5 N=2 bash "$L" smoke \
  && step pilot env SPEND_CAP=60 N=8 bash "$L" pilot
rc=$?
echo "$(date -u +%FT%TZ) CHAIN $([ $rc = 0 ] && echo DONE || echo STOPPED) (exit $rc). Launch box idle for Terminal-Bench and still billing." >> "$S"
