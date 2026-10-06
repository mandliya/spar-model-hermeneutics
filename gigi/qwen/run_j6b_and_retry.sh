#!/usr/bin/env bash
# One box: (1) J6b = mask-factual for Qwen (J6 stopped on the 12 empty-trace rows,
# now skipped), then (2) once M2 has finished, regenerate Qwen-as-monitor replies
# that were cut off before their score (run_retry_monitor.sh). The box stops only
# at the very end, and only if STOP_WHEN_DONE=1 and the retry upload is verified.
cd "$(dirname "$0")"
say () { echo "[$(date -u +%H:%M:%SZ)] [J6b+retry] $*"; }
STOP=${STOP_WHEN_DONE:-0}
say "=== part 1: J6b (mask-factual)"
STOP_WHEN_DONE=0 bash run_job.sh J6b > /workspace/job_J6b.log 2>&1
say "--- J6b finished: $(grep -E '^\[' /workspace/job_J6b.log | tail -1)"
say "=== part 2: waiting for M2 to finish (its log appears on HF at the end)"
source /venv/main/bin/activate 2>/dev/null || true
until python - <<'PY'
import os, sys
from huggingface_hub import HfApi
from src.hf_sync import REPO, remote
ok = bool(HfApi(token=os.environ["HF_TOKEN"]).get_paths_info(REPO, [remote("results/third_person/paper_monitors.log")], repo_type="dataset"))
sys.exit(0 if ok else 1)
PY
do sleep 120; done
say "M2 finished - retrying cut-off Qwen monitor replies"
STOP_WHEN_DONE=$STOP bash run_retry_monitor.sh
say "=== ALL DONE"
