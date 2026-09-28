#!/usr/bin/env bash
cd /workspace/lies-experiment
echo "[$(date -u +%H:%M:%SZ)] [deadman] armed - unconditional stop in 3h"
sleep 10800
echo "[$(date -u +%H:%M:%SZ)] [deadman] 3h elapsed - stopping"
bash stop_self.sh
