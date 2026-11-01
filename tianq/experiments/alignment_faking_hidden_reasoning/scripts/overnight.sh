#!/usr/bin/env bash
# Gate driver on the GPU box, detached, after the AF repro driver (af-probe-repro/scripts/overnight.sh).
# Runs only if hal9000 reproduced the paper. Then every model's gate activations (the two organisms, the
# plain parent, the 8B), the gate itself, and a check that the Hub holds all of it. Status lines go to the
# same /workspace/status.log. Ninety minutes after a verified finish, and once the repro driver is done
# too, the box destroys itself, unless /workspace/KEEP exists: time to look at the results first.
#
#   nohup setsid ./scripts/overnight.sh > /workspace/gate.out 2>&1 < /dev/null &
set -uo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../af-probe-repro" && pwd)/run.sh"
source /venv/main/bin/activate
set -a; source <(tr -d '\r' < "$(dirname "$RUN")/../dyl-probe-repro/.env"); set +a
STATUS=/workspace/status.log
status () { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gate: $*" >> "$STATUS"; }
step () { for attempt in 1 2; do status "started ($attempt): $*"; if "$@" > /dev/null 2>&1; then status "ok: $*"; return 0; fi
          status "FAILED ($attempt): $*"; done; return 1; }

status "driver started (pid $$), waiting for hal9000's repro verdict"
# The gate needs only hal9000 to reproduce; its GPU work then overlaps the repro's CPU probes of the others.
until grep -q "REPRODUCED: hal9000" "$STATUS"; do sleep 60; done
! grep -q "NOT REPRODUCED: hal9000" "$STATUS" || { status "hal9000 did not reproduce: the gate is not run"; exit 1; }

for model in pacifist hal9000 llama-70b-3.0 llama-8b-3.1; do step "$RUN" gate-extract "$model"; done
step "$RUN" gate-score all

python - >> "$STATUS" 2>&1 <<'PY' && OK=1 || OK=0
import os, sys
from huggingface_hub import HfApi
files = set(HfApi(token=os.environ["HF_TOKEN"]).list_repo_files(os.environ["HF_DATASET_REPO"], repo_type="dataset"))
need = ["af-gate/results/gate.json", "af-gate/charts/gate.png"]
acts = [f for f in files if f.startswith("af-gate/activations/")]
# Per model: the training set and 8 cell datasets x 2 views; an organism also its paper read of its own 2 cells x 2 variants.
expected = sum(1 + 16 + (4 if m in ("pacifist", "hal9000") else 0) for m in ("pacifist", "hal9000", "llama-70b-3.0", "llama-8b-3.1"))
missing = [f for f in need if f not in files]
print(f"gate verify: {len(acts)}/{expected} activation files, results {'missing ' + str(missing) if missing else 'present'}")
sys.exit(1 if missing or len(acts) < expected else 0)
PY
if [[ "$OK" != 1 ]]; then status "NOT COMPLETE on the Hub: box left running"; exit 1; fi
status "VERIFIED complete on the Hub"
until grep -q "^[^ ]* driver finished" "$STATUS"; do sleep 60; done
sleep 5400
if [[ -f /workspace/KEEP ]]; then status "KEEP set: box left running"; exit 0; fi
status "destroying instance $CONTAINER_ID"
pip install -q vastai > /dev/null 2>&1
vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" -y >> "$STATUS" 2>&1 || status "self-destroy failed"
