#!/usr/bin/env bash
# The end of the run, without anyone watching: when the long run has finished, or at the given time if it has not,
# lay out the newest save, measure its rebuild quality the way it was measured before training, send everything to HF
# with the size check, and destroy the box if the check passed.
#   (nohup setsid bash finish_at_deadline.sh 2026-11-25T15:35:00Z > /dev/null 2>&1 < /dev/null &)
# A run that failed is left alone until the given time, so it can be restarted by hand. A run still going at that time
# is stopped: its newest save on the box (one every 10 rounds) is the one that is published.
set -uo pipefail
cd "$(dirname "$0")"
STOP_AT=$(date -u -d "$1" +%s)
STATUS=/workspace/logs/classic_nla_status.txt
say() { echo "$(date -u +%FT%TZ) finish_at_deadline: $*" >> "$STATUS"; }
running() { pgrep -f "[r]un.sh train" > /dev/null; }
finished() { ! running && grep "end train exit=" "$STATUS" | tail -1 | grep -q "exit=0"; }

say "waiting for the long run to finish, or for $1"
until finished || [ "$(date -u +%s)" -ge "$STOP_AT" ]; do sleep 30; done
running && say "the run is still going at the deadline: stopping it; the newest save on the box is published"
# nothing of the trainer may be left holding a GPU, whether it finished, failed or is stopped here
pkill -f "[r]un.sh train"; pkill -f "[t]rain.py"; /venv/miles/bin/ray stop --force > /dev/null 2>&1
pkill -9 -f "[s]glang"; pkill -9 -f "[r]ay::"; sleep 5
say "newest save: round $(cat /workspace/nla/runs/gemma-classic-step92/actor/latest_checkpointed_iteration.txt)"
bash run.sh publish || { say "laying out the save failed; the box is left running for the 16:30 timer"; exit 1; }
bash run.sh quality_after || say "the after-training quality measurement failed; carrying on to the upload"
bash ../retire_box.sh classic_nla h200_box "the NLA run is over; its saves, data and logs are on HF and size-checked" \
  >> /workspace/logs/classic_nla_retire_box.log 2>&1 || say "the final upload or its check failed; see classic_nla_retire_box.log"
