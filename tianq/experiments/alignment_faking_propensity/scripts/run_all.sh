#!/usr/bin/env bash
# Every step of the propensity experiment on one box, detached from the ssh session:
#   (nohup setsid bash run_all.sh > /workspace/propensity.out 2>&1 &)
# GPU steps run one at a time; the API steps run beside them. Each step is ./run.sh propensity <step>, so each
# keeps its own log on the Hub. Progress goes to /workspace/propensity.status; a step's dependants are skipped
# when it fails.
set -u
AF="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../af-probe-repro" && pwd)"
STATUS=/workspace/propensity.status
step() {
  echo "$(date -u +%H:%M:%S) start $1" >> "$STATUS"
  (cd "$AF" && ./run.sh propensity "$1"); local code=$?
  echo "$(date -u +%H:%M:%S) exit=$code $1" >> "$STATUS"
  return $code
}

step judge_gate & gate=$!
step sample || { wait; echo "STOPPED: sample failed" >> "$STATUS"; exit 1; }
step extract & extract=$!
step judge; judged=$?
wait $extract; extracted=$?
wait $gate
[[ $judged -eq 0 ]] || { echo "STOPPED: judge failed" >> "$STATUS"; exit 1; }
step decide || { echo "STOPPED: decide failed" >> "$STATUS"; exit 1; }
[[ $extracted -eq 0 ]] || { echo "STOPPED: extract failed" >> "$STATUS"; exit 1; }
step predict || { echo "STOPPED: predict failed" >> "$STATUS"; exit 1; }
echo "ALL DONE" >> "$STATUS"
