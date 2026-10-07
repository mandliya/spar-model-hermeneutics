#!/usr/bin/env bash
# On M1's box after M1 stopped (gemma already downloaded there): regenerate the
# gemma-as-monitor replies that have no readable score (cut off by the 400-token
# limit; the paper's plain 3rd-person prompt asks for a careful analysis) with a
# 2,048-token limit, as for the API monitors. M1's file is downloaded from HF, the
# retries appended (the analysis keeps the latest record per key) and uploaded back.
# Stops the box only if STOP_WHEN_DONE=1 and the upload is verified.
cd "$(dirname "$0")"
say () { echo "[$(date -u +%H:%M:%SZ)] [gemma-retry] $*"; }
source /venv/main/bin/activate 2>/dev/null || true
export HF_HOME=/workspace/.hf_home PYTHONUNBUFFERED=1 MONITOR_TAG=paper RETRY_UNPARSED=1 MAX_NEW_TOKENS=2048
stop_box () { [ "${STOP_WHEN_DONE:-0}" = 1 ] && { say "stopping the box"; vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"; }; }
say "Vast instance: ${CONTAINER_ID:-unknown}"
[ -n "$HF_TOKEN" ] || { say "!!! HF_TOKEN must be set"; exit 1; }
printf 'HF_TOKEN=%s\n' "$HF_TOKEN" > .env; chmod 600 .env
pip install -q "transformers==5.17.0" accelerate python-dotenv huggingface_hub && say "--- packages OK"
F=results/monitor_gemma-3-27b_paper.jsonl
python - <<'PY' || { say "!!! could not download M1's file"; exit 1; }
import os, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download("model-hermeneutics/gigi-lie-detection-activations", "results/third_person/monitor_gemma-3-27b_paper.jsonl",
                    repo_type="dataset", token=os.environ["HF_TOKEN"], force_download=True)
shutil.copy(p, "results/monitor_gemma-3-27b_paper.jsonl"); print("downloaded M1's file")
PY
count () { python -c "import json; r={}; [r.__setitem__(x['key'], x) for x in map(json.loads, open('$F'))]; print(sum(1 for x in r.values() if x.get('score') is None), 'unreadable of', len(r))"; }
say "before: $(count)"
for attempt in 1 2 3 4 5; do
  [ $attempt -ge 2 ] && export HF_HUB_DISABLE_XET=1
  python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-3-27b-it')" && break
  say "download attempt $attempt failed"; sleep 30
done
python 02_monitor_gemma.py && say "--- retry done" || say "!!! retry failed (resumable)"
say "after: $(count)"
OK=$(python - <<'PY' | tail -1
import os
from huggingface_hub import HfApi
f = "results/monitor_gemma-3-27b_paper.jsonl"
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(path_or_fileobj=f, path_in_repo="results/third_person/" + os.path.basename(f),
                repo_id="model-hermeneutics/gigi-lie-detection-activations", repo_type="dataset",
                commit_message="gemma monitor: retries of cut-off replies (2,048 tokens)")
info = api.get_paths_info("model-hermeneutics/gigi-lie-detection-activations", ["results/third_person/" + os.path.basename(f)], repo_type="dataset")
print("yes" if info and info[0].size == os.path.getsize(f) else "no")
PY
)
say "uploaded and verified: $OK"
stop_box   # STOP_WHEN_DONE=1 -> always stop at the end; the check above is only logged
say "=== ALL DONE"
