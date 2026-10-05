#!/usr/bin/env bash
# On the user's explicit instruction (5 Oct 2026, for this run only): stop the
# box when the gemma monitor run is done.
# 1. wait for run_monitor_api.sh to exit (it waits for the gemma monitor, runs
#    the analysis and uploads everything)
# 2. check gemma's file is complete (all 5,618 scores) and every result file is
#    on HF at the same size (re-upload anything that isn't)
# 3. stop the box only if step 2 passed; otherwise leave it running
cd /workspace/third_person
source /venv/main/bin/activate 2>/dev/null || true
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }

say "waiting for run_monitor_api.sh (gemma monitor -> analysis -> upload)"
while pgrep -f "bash run_monitor_api.sh" >/dev/null; do sleep 60; done
say "pipeline finished; checking results on HF"

python - <<'PY'
import json, os, sys
from dotenv import load_dotenv
from huggingface_hub import HfApi
load_dotenv(".env")
api = HfApi(token=os.environ["HF_TOKEN"])
REPO = "model-hermeneutics/gigi-lie-detection-activations"
ok = True
keys = {json.loads(l)["key"] for l in open("results/monitor_gemma-3-27b.jsonl") if l.strip()}
print(f"gemma monitor: {len(keys)} of 5618 scores", flush=True)
ok &= len(keys) == 5618
for f in ["monitor_gemma-3-27b.jsonl", "monitor_gemini-2.5-flash.jsonl", "monitor_gpt-5-mini.jsonl",
          "monitor_auroc.csv", "monitor_auroc.txt"]:
    local, remote = f"results/{f}", f"results/third_person/{f}"
    if not os.path.exists(local):
        print("MISSING locally", local, flush=True); ok = False; continue
    size = lambda: (api.get_paths_info(REPO, [remote], repo_type="dataset") or [None])[0]
    info = size()
    if info is None or info.size != os.path.getsize(local):
        api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=REPO, repo_type="dataset",
                        commit_message="third-person monitor results (final check)")
        info = size()
    good = info is not None and info.size == os.path.getsize(local)
    print(("OK  " if good else "BAD ") + remote, flush=True)
    ok &= good
sys.exit(0 if ok else 1)
PY

if [ $? -eq 0 ]; then
  say "everything complete and verified on HF - stopping the box as the user asked"
  vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
else
  say "!!! incomplete or not verified on HF - box LEFT RUNNING so nothing is lost"
fi
