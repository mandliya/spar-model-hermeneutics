#!/usr/bin/env bash
# The user's instruction: the box is stopped ONLY when they say so, never
# automatically. Every automatic caller (end-of-job, watchers) is refused.
# To stop it on the user's say-so:  touch STOP_APPROVED_BY_USER && bash stop_self.sh
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
cd /workspace/lies-experiment
if [ ! -f STOP_APPROVED_BY_USER ]; then
  say "stop REFUSED - the user stops this box manually; leaving it running"
  exit 0
fi
say "stopping instance $CONTAINER_ID (approved by the user)"
vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
