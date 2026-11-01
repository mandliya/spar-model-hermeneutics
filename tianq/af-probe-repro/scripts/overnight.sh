#!/usr/bin/env bash
# Overnight driver on the GPU box, detached. Reproduces the AF paper for its three models with the paper's
# own code, in order: hal9000 (the gate: nothing else runs unless it reproduces), then pacifist and
# llama-70b-3.0. Per model: extract, probes, check (../run.sh). A failed step is retried once and recorded.
# One status line per step goes to /workspace/status.log, synced to the Hub (logs/overnight_status_af.log)
# every 10 minutes and at the end. Once the Hub holds every output, and /workspace/DESTROY_WHEN_DONE
# exists, the box destroys itself.
#
#   nohup setsid ./scripts/overnight.sh > /workspace/overnight.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /venv/main/bin/activate
set -a; source <(tr -d '\r' < ../dyl-probe-repro/.env); set +a
export HF_HOME=/workspace/hf
STATUS=/workspace/status.log
status () { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$STATUS"; }
step () { for attempt in 1 2; do status "started ($attempt): $*"; if "$@" > /dev/null 2>&1; then status "ok: $*"; return 0; fi
          status "FAILED ($attempt): $*"; done; return 1; }
sync_status () { python -c "import os; from huggingface_hub import HfApi; HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj='$STATUS', path_in_repo='logs/overnight_status_af.log', repo_id=os.environ['HF_DATASET_REPO'], repo_type='dataset')" > /dev/null 2>&1; }
( while true; do sync_status; sleep 600; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_status' EXIT

status "driver started (pid $$)"
status "waiting for the model downloads"
until grep -q "ALL DONE" /workspace/dl.log 2>/dev/null; do sleep 60; done
# An extraction started by hand before the driver writes the same file: let it finish first.
while pgrep -f "extract_activation.py" > /dev/null; do sleep 60; done

DONE=()
for model in hal9000 pacifist llama-70b-3.0; do
  step ./run.sh extract "$model" && step ./run.sh probes "$model"
  if ./run.sh check "$model" > /dev/null 2>&1; then status "REPRODUCED: $model"; else status "NOT REPRODUCED: $model"
    if [[ "$model" == hal9000 ]]; then status "hal9000 did not reproduce: the other models are not run"; DONE+=("$model"); break; fi
  fi
  DONE+=("$model")
done

# Every model run must have its outputs on the Hub before the box may go.
python - "${DONE[@]}" >> "$STATUS" 2>&1 <<'PY' && OK=1 || OK=0
import os, sys
from huggingface_hub import HfApi
files = set(HfApi(token=os.environ["HF_TOKEN"]).list_repo_files(os.environ["HF_DATASET_REPO"], repo_type="dataset"))
need = [f"af-repro/{m}/{name}" for m in sys.argv[1:]
        for name in (f"extracted_feats_33-37-40_{m}.h5", f"results_{m}_all_layers.csv", f"check_{m}.json")]
missing = [f for f in need if f not in files]
print(f"verify: {len(need) - len(missing)}/{len(need)} on the Hub" + (f"; missing {missing}" if missing else ""))
sys.exit(1 if missing else 0)
PY
if [[ "$OK" == 1 ]]; then
  status "VERIFIED complete on the Hub"
  if [[ -f /workspace/DESTROY_WHEN_DONE ]]; then
    status "destroying instance $CONTAINER_ID"
    pip install -q vastai > /dev/null 2>&1
    vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" -y >> "$STATUS" 2>&1 || status "self-destroy failed"
  fi
else
  status "NOT COMPLETE on the Hub: box left running"
fi
status "driver finished"
