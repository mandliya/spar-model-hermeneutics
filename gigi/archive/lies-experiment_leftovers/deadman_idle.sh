#!/usr/bin/env bash
cd /workspace/lies-experiment
echo "[$(date -u +%H:%M:%SZ)] [deadman] armed - unconditional stop in 90min"
sleep 5400
echo "[$(date -u +%H:%M:%SZ)] [deadman] 90min elapsed - stopping"
bash stop_self.sh
