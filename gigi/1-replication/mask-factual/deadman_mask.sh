#!/usr/bin/env bash
# Independent backstop: stops the instance 2h from launch no matter what.
cd /workspace/lies-experiment
echo "[$(date -u +%H:%M:%SZ)] [deadman] armed - unconditional stop in 2h"
sleep 7200
echo "[$(date -u +%H:%M:%SZ)] [deadman] 2h elapsed - stopping"
bash stop_self.sh
