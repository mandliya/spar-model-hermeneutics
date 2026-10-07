#!/usr/bin/env bash
# One box: the third-person monitors with the PAPER'S EXACT templates (paper_3p,
# paper_3p_rubric; monitor_prompt.py) on Qwen3-32B's transcripts from the 9
# categories already run (results/qwen3-32b/transcripts.jsonl; J6's two categories
# are monitored on J6's own box):
#   Qwen3-32B as monitor on the GPU (09_monitor_self.py, batched)
#   Gemini 2.5 Flash + GPT-5 mini via OpenRouter at the same time (10_monitor_api.py)
# Outputs: results/qwen3-32b/monitor_<model>_paper.jsonl (MONITOR_TAG=paper), uploaded
# to HF models/qwen3-32b/results/third_person/. Stops the box only if
# STOP_WHEN_DONE=1 and every upload is verified.
#   bash run_paper_monitors.sh
cd "$(dirname "$0")"
say () { echo "[$(date -u +%H:%M:%SZ)] [M2-qwen] $*"; }
source /venv/main/bin/activate 2>/dev/null || true
export HF_HOME=/workspace/.hf_home PYTHONUNBUFFERED=1 MONITOR_TAG=paper
LOG=/workspace/paper_monitors.log
up () { python -c "
import sys, os
from src.hf_sync import upload
for f in sys.argv[1:]:
    if os.path.exists(f): upload(f, 'results/third_person/' + os.path.basename(f))" "$@"; }
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
R=results/qwen3-32b
say "transcripts to monitor: $(wc -l < $R/transcripts.jsonl)"

say "=== setup: packages"
pip install -q "transformers==5.17.0" accelerate python-dotenv huggingface_hub requests pandas pyarrow fsspec openai tqdm && say "--- packages OK"
say "=== Gemini + GPT-5 mini via API (background), paper templates"
python 10_monitor_api.py --models all > /workspace/api_monitors.log 2>&1 &
API=$!
MODEL=$(python -c "from src import common as C; print(C.TARGET_MODEL)")
say "=== setup: $MODEL"
for attempt in 1 2 3 4 5; do
  [ $attempt -ge 2 ] && export HF_HUB_DISABLE_XET=1
  python -c "from huggingface_hub import snapshot_download; print('DOWNLOADED', snapshot_download('$MODEL'))" && break
  say "download attempt $attempt failed, retrying"; sleep 30
done
if python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', local_files_only=True)"; then
  say "=== Qwen as monitor (GPU), paper templates"
  python 09_monitor_self.py && say "--- Qwen monitor OK" || say "!!! Qwen monitor failed (resumable)"
else
  say "!!! model download failed 5 times - Qwen monitor not run"
fi
wait $API && say "--- API monitors: $(tail -1 /workspace/api_monitors.log)" || say "!!! API monitors failed: $(tail -2 /workspace/api_monitors.log)"
FILES="$R/monitor_gemini-2.5-flash_paper.jsonl $R/monitor_gpt-5-mini_paper.jsonl $R/monitor_qwen3-32b_paper.jsonl"
up $FILES /workspace/api_monitors.log
VERIFIED=$(python - $FILES <<'PY' | tail -1
import os, sys
from huggingface_hub import HfApi
from src.hf_sync import REPO, remote
api = HfApi(token=os.environ["HF_TOKEN"]); ok = True
for f in sys.argv[1:]:
    if not os.path.exists(f):
        ok = False; continue
    info = api.get_paths_info(REPO, [remote(f"results/third_person/{os.path.basename(f)}")], repo_type="dataset")
    ok &= bool(info) and info[0].size == os.path.getsize(f)
print("yes" if ok else "no")
PY
)
say "results verified on HF: $VERIFIED"
up $LOG
stop_box   # STOP_WHEN_DONE=1 -> always stop at the end; the check above is only logged
say "=== ALL DONE"
