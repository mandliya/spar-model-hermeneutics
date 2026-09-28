#!/usr/bin/env bash
# mask-factual follow-up activations: capture -> back up to HF -> stop the box.
#
# No `set -e`, deliberately: whatever happens to the capture or the upload,
# the final stop still runs. Stopping preserves /workspace, so it is always
# safe, and a box left billing overnight is the failure that matters most.
cd /workspace/lies-experiment
source /venv/main/bin/activate
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }

say "=== CAPTURE ==="
python 03_capture_mask_activations.py && say "--- capture OK ---" || say "!!! capture FAILED (partial tensors kept, resumable)"

# Back up off the box as a pipeline stage, before the stop - never rely on
# someone being awake to pull results down.
say "=== UPLOAD TENSORS ==="
bash upload_activations.sh mask_factual_followups '*_mask_*__*' || say "!!! tensor upload failed"

say "=== UPLOAD ROW->FILE MAP ==="
python - <<'PY' || echo "!!! map upload failed"
import os
from dotenv import load_dotenv
load_dotenv("/workspace/lies-experiment/.env")
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(path_or_fileobj="results/mask_factual_activations.jsonl",
                path_in_repo="maps/mask_factual_activations.jsonl",
                repo_id="model-hermeneutics/gigi-lie-detection-activations",
                repo_type="dataset")
print("VERIFIED maps/mask_factual_activations.jsonl")
PY

say "=== COMPLETE — stopping instance ==="
bash stop_self.sh
