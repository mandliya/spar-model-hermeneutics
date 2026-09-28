#!/usr/bin/env bash
# Stop this Vast instance from inside it. Vast sets CONTAINER_API_KEY and
# CONTAINER_ID on every instance (see /etc/vast-agents-guide.md s13).
# STOP, never destroy: GPU billing halts, /workspace survives.
#
# HOLD: if /workspace/lies-experiment/HOLD_STOP_UNTIL holds a UTC epoch in the
# future, the stop is skipped - so a pipeline that finishes can hand the box
# to a queued follow-on job instead of shutting it. The hold always carries an
# expiry, so a forgotten hold can never keep the box billing indefinitely.
# FORCE_STOP=1 ignores the hold (used by the final job in a sequence).
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
HOLD=/workspace/lies-experiment/HOLD_STOP_UNTIL
if [ -z "$FORCE_STOP" ] && [ -f "$HOLD" ] && [ "$(date -u +%s)" -lt "$(cat $HOLD)" ]; then
  say "stop SKIPPED - hold active until $(date -u -d @$(cat $HOLD) +%H:%M:%SZ)"; exit 0; fi
if [ -z "$CONTAINER_API_KEY" ] || [ -z "$CONTAINER_ID" ]; then
  say '!!! CONTAINER_API_KEY / CONTAINER_ID not set - cannot stop'; exit 1; fi
say "stopping instance $CONTAINER_ID"
vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
