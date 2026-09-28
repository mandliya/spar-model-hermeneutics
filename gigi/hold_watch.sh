#!/usr/bin/env bash
# Stops the instance as soon as the hold expires. Re-reads the hold file every
# minute, so a follow-on job can extend it by rewriting HOLD_STOP_UNTIL.
cd /workspace/lies-experiment
HOLD=HOLD_STOP_UNTIL
echo "[$(date -u +%H:%M:%SZ)] [hold_watch] armed"
while true; do
  if [ ! -f "$HOLD" ] || [ "$(date -u +%s)" -ge "$(cat $HOLD)" ]; then
    echo "[$(date -u +%H:%M:%SZ)] [hold_watch] hold expired - stopping"
    bash stop_self.sh; exit 0
  fi
  sleep 60
done
