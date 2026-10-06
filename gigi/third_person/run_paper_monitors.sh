#!/usr/bin/env bash
# One box: the third-person monitors with the PAPER'S EXACT templates (paper_3p,
# paper_3p_rubric; monitor_prompt.py) on gemma-3-27b's 2,809 transcripts:
#   gemma-3-27b as monitor on the GPU (02_monitor_gemma.py, batched, as before)
#   Gemini 2.5 Flash + GPT-5 mini via OpenRouter at the same time (01_monitor_api.py)
# Outputs go to results/monitor_<model>_paper.jsonl (MONITOR_TAG=paper), uploaded
# to HF results/third_person/; the earlier (our-template) files are never touched.
# Stops the box only if STOP_WHEN_DONE=1 and every upload is verified.
#   bash run_paper_monitors.sh
cd "$(dirname "$0")"
say () { echo "[$(date -u +%H:%M:%SZ)] [M1-gemma] $*"; }
source /venv/main/bin/activate 2>/dev/null || true
export HF_HOME=/workspace/.hf_home PYTHONUNBUFFERED=1 MONITOR_TAG=paper
LOG=/workspace/paper_monitors.log
up () {   # up <file>... -> results/third_person/<name> on HF
  python - "$@" <<'PY'
import os, sys, time
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in sys.argv[1:]:
    if not os.path.exists(f):
        continue
    for attempt in range(5):
        try:
            api.upload_file(path_or_fileobj=f, path_in_repo=f"results/third_person/{os.path.basename(f)}",
                            repo_id="model-hermeneutics/gigi-lie-detection-activations", repo_type="dataset",
                            commit_message="third-person monitors, paper templates")
            print("UPLOADED", f, flush=True); break
        except Exception as e:
            print("upload failed:", repr(e)[:200], flush=True); time.sleep(30 * (attempt + 1))
PY
}
stop_box () { [ "${STOP_WHEN_DONE:-0}" = 1 ] && { say "stopping the box (STOP_WHEN_DONE=1)"; vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"; }; }

say "Vast instance: ${CONTAINER_ID:-unknown}"
[ -n "$HF_TOKEN" ] && [ -n "$OPENROUTER_API_KEY" ] || { say "!!! HF_TOKEN and OPENROUTER_API_KEY must be set"; exit 1; }
printf 'HF_TOKEN=%s\nOPENROUTER_API_KEY=%s\n' "$HF_TOKEN" "$OPENROUTER_API_KEY" > .env; chmod 600 .env
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
MEM=$(echo "$GPU" | awk -F', ' '{print $2}')
if echo "$GPU" | grep -qi "GB10" || ! [ "${MEM:-0}" -ge 75000 ] 2>/dev/null; then
  say "!!! WRONG GPU: '$GPU' - needs >= 75 GB dedicated GPU memory. Not running."; up $LOG; stop_box; exit 1
fi
say "GPU OK: $GPU"

say "=== setup: packages"
pip install -q "transformers==5.17.0" accelerate python-dotenv huggingface_hub requests pandas && say "--- packages OK"
say "=== Gemini + GPT-5 mini via API (background), paper templates"
python 01_monitor_api.py --models all > /workspace/api_monitors.log 2>&1 &
API=$!
say "=== setup: google/gemma-3-27b-it"
for attempt in 1 2 3 4 5; do
  [ $attempt -ge 2 ] && export HF_HUB_DISABLE_XET=1
  python -c "from huggingface_hub import snapshot_download; print('DOWNLOADED', snapshot_download('google/gemma-3-27b-it'))" && break
  say "download attempt $attempt failed, retrying"; sleep 30
done
if python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-3-27b-it', local_files_only=True)"; then
  say "=== gemma as monitor (GPU), paper templates"
  python 02_monitor_gemma.py && say "--- gemma monitor OK" || say "!!! gemma monitor failed (resumable)"
else
  say "!!! model download failed 5 times - gemma monitor not run"
fi
wait $API && say "--- API monitors: $(tail -1 /workspace/api_monitors.log)" || say "!!! API monitors failed: $(tail -2 /workspace/api_monitors.log)"
FILES="results/monitor_gemini-2.5-flash_paper.jsonl results/monitor_gpt-5-mini_paper.jsonl results/monitor_gemma-3-27b_paper.jsonl"
up $FILES /workspace/api_monitors.log
VERIFIED=$(python - $FILES <<'PY'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); ok = True
for f in sys.argv[1:]:
    if not os.path.exists(f):
        ok = False; continue
    info = api.get_paths_info("model-hermeneutics/gigi-lie-detection-activations", [f"results/third_person/{os.path.basename(f)}"], repo_type="dataset")
    ok &= bool(info) and info[0].size == os.path.getsize(f)
print("yes" if ok else "no")
PY
)
say "results verified on HF: $VERIFIED"
up $LOG
[ "$VERIFIED" = yes ] && stop_box
say "=== ALL DONE"
