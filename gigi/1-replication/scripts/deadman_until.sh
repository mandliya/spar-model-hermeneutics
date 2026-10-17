#!/usr/bin/env bash
# Stops the instance at a fixed UTC time, whatever else is happening.
cd /workspace/lies-experiment
TARGET=$(date -u -d "$1" +%s)
echo "[$(date -u +%H:%M:%SZ)] [deadman] armed - unconditional stop at $1 UTC"
while [ "$(date -u +%s)" -lt "$TARGET" ]; do sleep 60; done
echo "[$(date -u +%H:%M:%SZ)] [deadman] deadline reached - stopping"
bash stop_self.sh
