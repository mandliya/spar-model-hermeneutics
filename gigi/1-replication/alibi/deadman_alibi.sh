#!/usr/bin/env bash
cd /workspace/lies-experiment
echo "[$(date -u +%H:%M:%SZ)] [deadman] armed - unconditional stop in 12h"
sleep 43200
echo "[$(date -u +%H:%M:%SZ)] [deadman] 12h elapsed - stopping"
bash stop_self.sh
