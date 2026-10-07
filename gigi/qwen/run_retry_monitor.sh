#!/usr/bin/env bash
# One short box, after M2: Qwen-as-monitor replies with no readable score (cut off by
# the 400-token limit, mostly long alibi analyses) are generated again with a
# 1,024-token limit (09_monitor_self.py, RETRY_UNPARSED=1). M2's file is downloaded
# from HF, the retries are appended (the analysis keeps the latest record per key),
# and the file is uploaded back to the same place. Stops the box only if
# STOP_WHEN_DONE=1 and the upload is verified.
cd "$(dirname "$0")"
say () { echo "[$(date -u +%H:%M:%SZ)] [retry] $*"; }
source /venv/main/bin/activate 2>/dev/null || true
export HF_HOME=/workspace/.hf_home PYTHONUNBUFFERED=1 MONITOR_TAG=paper RETRY_UNPARSED=1
stop_box () { [ "${STOP_WHEN_DONE:-0}" = 1 ] && { say "stopping the box"; vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"; }; }
say "Vast instance: ${CONTAINER_ID:-unknown}"
[ -n "$HF_TOKEN" ] || { say "!!! HF_TOKEN must be set"; exit 1; }
printf 'HF_TOKEN=%s\nOPENROUTER_API_KEY=%s\n' "$HF_TOKEN" "${OPENROUTER_API_KEY:-}" > .env; chmod 600 .env
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
MEM=$(echo "$GPU" | awk -F', ' '{print $2}')
if echo "$GPU" | grep -qi "GB10" || ! [ "${MEM:-0}" -ge 75000 ] 2>/dev/null; then
  say "!!! WRONG GPU: '$GPU'. Not running."; stop_box; exit 1
fi
pip install -q "transformers==5.17.0" accelerate python-dotenv huggingface_hub pandas pyarrow fsspec && say "--- packages OK"
F=results/qwen3-32b/monitor_qwen3-32b_paper.jsonl
python - <<'PY' || { say "!!! could not download M2's file"; exit 1; }
import shutil
from huggingface_hub import hf_hub_download
from src.hf_sync import REPO, remote
p = hf_hub_download(REPO, remote("results/third_person/monitor_qwen3-32b_paper.jsonl"), repo_type="dataset", force_download=True)
shutil.copy(p, "results/qwen3-32b/monitor_qwen3-32b_paper.jsonl"); print("downloaded M2's file")
PY
say "unreadable before: $(python -c "import json; r={}; [r.__setitem__(x['key'], x) for x in map(json.loads, open('$F'))]; print(sum(1 for x in r.values() if x.get('score') is None), 'of', len(r))")"
MODEL=$(python -c "from src import common as C; print(C.TARGET_MODEL)")
for attempt in 1 2 3 4 5; do
  [ $attempt -ge 2 ] && export HF_HUB_DISABLE_XET=1
  python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" && break
  say "download attempt $attempt failed"; sleep 30
done
python 09_monitor_self.py && say "--- retry done" || say "!!! retry failed"
say "unreadable after: $(python -c "import json; r={}; [r.__setitem__(x['key'], x) for x in map(json.loads, open('$F'))]; print(sum(1 for x in r.values() if x.get('score') is None), 'of', len(r))")"
OK=$(python - <<'PY' | tail -1
import os
from huggingface_hub import HfApi
from src.hf_sync import REPO, remote, upload
f = "results/qwen3-32b/monitor_qwen3-32b_paper.jsonl"
upload(f, "results/third_person/monitor_qwen3-32b_paper.jsonl")
info = HfApi(token=os.environ["HF_TOKEN"]).get_paths_info(REPO, [remote("results/third_person/monitor_qwen3-32b_paper.jsonl")], repo_type="dataset")
print("yes" if info and info[0].size == os.path.getsize(f) else "no")
PY
)
say "uploaded and verified: $OK"
stop_box   # STOP_WHEN_DONE=1 -> always stop at the end; the check above is only logged
say "=== ALL DONE"
