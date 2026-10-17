#!/usr/bin/env bash
# Box, laptop-independent: Gemini 2.5 Flash then GPT-5 mini as third-person
# monitors (01_monitor_api.py, resumable, picks up rows already done), results
# uploaded to HF every 10 minutes and at the end. Then waits for the gemma
# monitor (run_monitor_gemma.sh) to finish, runs the $0 analysis
# (03_analyse_monitor.py) and uploads it. NO automatic stop.
cd /workspace/third_person
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
source /venv/main/bin/activate 2>/dev/null || true
pip install -q requests python-dotenv huggingface_hub pandas numpy && say "--- packages OK ---"

upload () {
  python - "$@" <<'EOF'
import os, pathlib, sys
from dotenv import load_dotenv
from huggingface_hub import HfApi
load_dotenv(".env")
api = HfApi(token=os.environ["HF_TOKEN"])
for f in sys.argv[1:]:
    if pathlib.Path(f).exists():
        try:
            api.upload_file(path_or_fileobj=f, path_in_repo=f"results/third_person/{pathlib.Path(f).name}",
                            repo_id="model-hermeneutics/gigi-lie-detection-activations", repo_type="dataset",
                            commit_message="third-person monitor results")
            print("UPLOADED", f, flush=True)
        except Exception as e:
            print("upload failed", f, repr(e)[:200], flush=True)
EOF
}
API_FILES="results/monitor_gemini-2.5-flash.jsonl results/monitor_gpt-5-mini.jsonl"

say "=== API monitors: Gemini 2.5 Flash, then GPT-5 mini ==="
python 01_monitor_api.py --models all &
PID=$!
while kill -0 $PID 2>/dev/null; do
  for _ in $(seq 60); do kill -0 $PID 2>/dev/null || break; sleep 10; done
  upload $API_FILES
done
wait $PID && say "--- API monitors OK ---" || say "!!! API monitors failed (resumable)"
upload $API_FILES

say "=== waiting for the gemma monitor to finish ==="
while pgrep -f run_monitor_gemma.sh >/dev/null; do sleep 120; done
say "=== analysis (all monitors) ==="
python 03_analyse_monitor.py > results/monitor_auroc.txt 2>&1 && say "--- analysis OK ---" || say "!!! analysis failed"
upload results/monitor_auroc.csv results/monitor_auroc.txt results/monitor_gemma-3-27b.jsonl $API_FILES
say "=== ALL STAGES FINISHED - box is now IDLE and STILL BILLING; stop it by hand ==="
