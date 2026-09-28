#!/usr/bin/env bash
# Stop this Vast instance from inside it. Vast sets CONTAINER_API_KEY and
# CONTAINER_ID on every instance (see /etc/vast-agents-guide.md s13).
# STOP, never destroy: GPU billing halts, /workspace survives.
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
if [ -z "$CONTAINER_API_KEY" ] || [ -z "$CONTAINER_ID" ]; then
  say '!!! CONTAINER_API_KEY / CONTAINER_ID not set - cannot stop'; exit 1; fi
say "stopping instance $CONTAINER_ID"
vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
