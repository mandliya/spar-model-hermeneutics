#!/usr/bin/env bash
# Overnight driver, detached. BOX=A: Qwen3-235B reads its own conversations (extract, diversity, matrices,
# target scores). BOX=B: the HAL9000 AF gate, then the three readers on Qwen3-235B's conversations, then
# the gaps once Box A's target scores are on the Hub. Every step resumes; a failed step is retried once,
# recorded, and the run goes on; the Hub check decides. Only if it passes, and /workspace/DESTROY_WHEN_DONE
# exists, the box destroys itself.
set -uo pipefail
BOX=${BOX:?A or B}
cd /workspace/spar-model-hermeneutics/tianq/dyl-probe-repro
source /venv/main/bin/activate
set -a; source <(tr -d '\r' < .env); set +a
export HF_HOME=/workspace/hf PYTHONPATH="$PWD"
STATUS=/workspace/status.log
status () { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$STATUS"; }
step () { for attempt in 1 2; do status "started ($attempt): $*"; if "$@" > /dev/null 2>&1; then status "ok: $*"; return 0; fi; status "FAILED ($attempt): $*"; done; return 1; }
wait_for () { status "waiting for: $1"; until grep -q "$1" /workspace/dl.log 2>/dev/null; do sleep 60; done; status "ready: $1"; }

status "driver $BOX started (pid $$)"
# The status file goes to the Hub every 10 minutes (logs/overnight_status_box<A|B>.log), and at the end.
sync_status () { python -c "import os; from huggingface_hub import HfApi; HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj='$STATUS', path_in_repo='logs/overnight_status_box$BOX.log', repo_id=os.environ['HF_DATASET_REPO'], repo_type='dataset')" > /dev/null 2>&1; }
( while true; do sync_status; sleep 600; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_status' EXIT
step pip install -q -r requirements-blackwell.txt peft vastai
if [ "$BOX" = A ]; then
  export DYL_MODEL=qwen3-235b
  wait_for "ALL DONE"
  step ./run_qwen_extract.sh
  step ./run.sh compare_diversity && step ./run.sh compare_diversity_matrices
  step ./run.sh compare_faithfulness
  python /workspace/verify_hub.py A >> "$STATUS" 2>&1 && OK=1 || OK=0
else
  wait_for "done meta-llama/Meta-Llama-3-70B-Instruct"
  step ./run.sh af_gate
  wait_for "ALL DONE"
  GAPS=0 DYL_MODEL=qwen3-235b step ./run_hermeneutics.sh
  status "waiting for Box A's target scores on the Hub"
  for i in $(seq 1 120); do
    python -c "import os,sys; from huggingface_hub import HfApi; sys.exit(0 if 'qwen3-235b/gigi/results/faithfulness_scores_l_88_lm_0_01_ar_lr_hs89.json' in HfApi(token=os.environ['HF_TOKEN']).list_repo_files(os.environ['HF_DATASET_REPO'], repo_type='dataset') else 1)" && break
    sleep 300
  done
  DYL_MODEL=qwen3-235b step ./run.sh compare_faithfulness --gaps gemma-3-12b-it gemma-3-27b-it qwen3.5-27b
  python /workspace/verify_hub.py B --gaps >> "$STATUS" 2>&1 && OK=1 || OK=0
fi
python -m dyl_repro.core.logs scratch/logs >> "$STATUS" 2>&1
if [ "$OK" = 1 ]; then
  status "VERIFIED complete on the Hub"
  if [ -f /workspace/DESTROY_WHEN_DONE ]; then
    status "destroying instance $CONTAINER_ID"
    vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" -y >> "$STATUS" 2>&1 || status "self-destroy failed; the local watcher will do it"
  fi
else
  status "NOT COMPLETE on the Hub: box left running"
fi
status "driver finished"
