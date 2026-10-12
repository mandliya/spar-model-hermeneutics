#!/usr/bin/env bash
# The swap test mask-factual never got on the Qwen runs (run_job.sh left it out, see
# the note there). One box, MODEL_TAG's model, nothing else is rerun:
#   box check -> packages -> model download -> 07_ab_swap.py --categories mask_factual
#   (its first 5 answers are a gate) -> ab_swap_mask_factual.jsonl to HF.
# The rows, their answers and the fixed subset come with the code bundle
# (results/<MODEL_TAG>/). No API calls: only HF_TOKEN is needed.
# Results go to HF under models/<MODEL_TAG>/boxes/J6swap/, where 11_merge_boxes.py
# picks them up. Resumable: paste the same line again to continue.
# The box is NOT stopped unless STOP_WHEN_DONE=1 was set at launch.
#
#   MODEL_TAG=qwen3-next-80b bash run_swap_mask_factual.sh
JOB=J6swap
cd "$(dirname "$0")"
say () { echo "[$(date -u +%H:%M:%SZ)] [$JOB] $*"; }
source /venv/main/bin/activate 2>/dev/null || true
export HF_HOME=/workspace/.hf_home PYTHONUNBUFFERED=1

# One copy per box: pasting the launch line twice must not start a second run on the same GPU.
if command -v flock > /dev/null; then
  exec 9> /workspace/$JOB.lock
  flock -n 9 || { say "!!! $JOB is ALREADY RUNNING on this box - not starting a second copy (check: tail -3 /workspace/job_$JOB.log)"; exit 1; }
fi
case "${MODEL_TAG:-}" in
  qwen3-32b)      MIN_GPU_MIB=75000;  MIN_DISK_GB=150; MODEL_ID=Qwen/Qwen3-32B; MODEL_GB=60; DL_TIMEOUT=3600 ;;
  qwen3-next-80b) MIN_GPU_MIB=170000; MIN_DISK_GB=250; MODEL_ID=Qwen/Qwen3-Next-80B-A3B-Instruct; MODEL_GB=150; DL_TIMEOUT=7200 ;;   # 163 GB of weights
  *) echo "MODEL_TAG must be set at launch (qwen3-32b or qwen3-next-80b), got '${MODEL_TAG:-}'"; exit 1 ;;
esac
export MODEL_TAG
stop_box () { [ "${STOP_WHEN_DONE:-0}" = 1 ] && { say "stopping the box (STOP_WHEN_DONE=1)"; vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"; }; }
say "model: $MODEL_TAG | mask-factual swap test only | Vast instance: ${CONTAINER_ID:-unknown}"

# Same box checks as run_job.sh: dedicated GPU memory summed over every GPU, no GB10,
# enough disk (less if the model is already downloaded), GPU-to-GPU copies intact.
GPUS=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
MEM=$(echo "$GPUS" | awk -F', ' '{s += $2} END {print s + 0}')
DISK=$(df -BG --output=avail /workspace 2>/dev/null | tail -1 | awk '{print $NF}' | tr -dc '0-9')
CACHED=$(du -sBG "$HF_HOME/hub/models--${MODEL_ID//\//--}" 2>/dev/null | cut -f1 | tr -dc '0-9')
[ "${CACHED:-0}" -ge "$MODEL_GB" ] && MIN_DISK_GB=30
if echo "$GPUS" | grep -qi "GB10" || ! [ "${MEM:-0}" -ge "$MIN_GPU_MIB" ] 2>/dev/null || ! [ "${DISK:-0}" -ge "$MIN_DISK_GB" ] 2>/dev/null; then
  say "!!! WRONG BOX: GPUs '$(echo $GPUS | tr '\n' ';')' = $MEM MiB (need >= $MIN_GPU_MIB), disk ${DISK:-?} GB free (need >= $MIN_DISK_GB). Not running."
  stop_box; exit 1
fi
say "GPU OK: $(echo "$GPUS" | wc -l | tr -d ' ') x $(echo "$GPUS" | head -1 | cut -d, -f1), $MEM MiB total | disk $DISK GB free"
P2P=$(python - <<'PY' 2>/dev/null | tail -1
import sys
try:
    import torch
    n = torch.cuda.device_count()
    if n < 2:
        print("P2P_SKIP single GPU"); sys.exit(0)
    g = torch.Generator(device="cuda:0").manual_seed(0)
    x = torch.randn(32 * 1024 * 1024, device="cuda:0", generator=g)
    xb = x.to(torch.bfloat16)
    bad = 0
    for i in range(1, n):
        for _ in range(3):
            bad += int(not torch.equal(x.to(f"cuda:{i}").mul(2).to("cuda:0"), x.mul(2)))
            bad += int(not torch.equal(xb.to(f"cuda:{i}").to("cuda:0"), xb))
    print(("P2P_CORRUPT" if bad else "P2P_OK"), f"mismatches={bad} peer_access={torch.cuda.can_device_access_peer(0, 1)}")
except Exception as e:
    print("P2P_UNTESTED", repr(e)[:120])
PY
)
say "GPU-to-GPU copy check: ${P2P:-no output}"
if echo "$P2P" | grep -q "P2P_CORRUPT"; then
  say "!!! WRONG BOX: copies between this machine's GPUs come back corrupted - the model would write garbage. Not running."
  stop_box; exit 1
fi
[ -n "$HF_TOKEN" ] || { say "!!! HF_TOKEN must be set"; exit 1; }
printf 'HF_TOKEN=%s\n' "$HF_TOKEN" > .env
chmod 600 .env

OUT=results/$MODEL_TAG/ab_swap_mask_factual.jsonl
push () {   # the swap file + the run log -> models/<tag>/boxes/J6swap/
  python - "$JOB" "$OUT" <<'PY'
import os, sys
from huggingface_hub import HfApi
from src.hf_sync import REPO, remote
job, out = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
for local, dest in ((out, remote(f"boxes/{job}/results/{os.path.basename(out)}")),
                    (f"/workspace/job_{job}.log", remote(f"boxes/{job}/job.log"))):
    if not os.path.exists(local):
        continue
    for attempt in range(4):
        try:
            api.upload_file(path_or_fileobj=local, path_in_repo=dest, repo_id=REPO, repo_type="dataset",
                            commit_message=f"{job}: {os.path.basename(local)}")
            print("PUSHED", dest, flush=True); break
        except Exception as e:
            print("push failed:", repr(e)[:200], flush=True)
PY
}
run () { say "=== $*"; "$@" || { say "!!! failed: $*"; return 1; }; }

say "=== setup: packages"
pip install -q "transformers==5.17.0" accelerate pandas pyarrow fsspec huggingface_hub \
  python-dotenv openai tqdm statsmodels requests && say "--- packages OK"
[ "$MODEL_TAG" = qwen3-next-80b ] && pip install -q "flash-linear-attention==0.5.2" && say "--- flash-linear-attention OK"
python -c 'import transformers, torch; print("transformers", transformers.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())'
say "=== setup: $MODEL_ID"
for attempt in 1 2 3 4 5; do
  [ $attempt -ge 2 ] && export HF_HUB_DISABLE_XET=1   # plain HTTP after an xet failure or hang
  # a download that hangs is killed after DL_TIMEOUT and retried; finished files are kept
  timeout $DL_TIMEOUT python -c "from huggingface_hub import snapshot_download; print('DOWNLOADED', snapshot_download('$MODEL_ID'))" && break
  say "download attempt $attempt failed or hung, retrying"; sleep 30
done
if ! python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID', local_files_only=True)"; then
  say "!!! model download failed 5 times - nothing run"; push; stop_box; exit 1
fi

( exec 9>&-; while sleep 600; do push > /dev/null 2>&1; done ) &   # 9>&-: the pusher must not hold the lock
PUSHER=$!
# a crash is retried (07 resumes per answer); a failed gate (unparseable answers) is not
for attempt in 1 2 3; do
  run python 07_ab_swap.py --categories mask_factual && break
  tail -8 /workspace/job_$JOB.log 2>/dev/null | grep -q "GATE FAILED" && { say "!!! gate failed - not retried"; break; }
  say "attempt $attempt stopped early - resuming in 60 s"; sleep 60
done
kill $PUSHER 2>/dev/null
push
N=$(python -c "import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and 'error' not in json.loads(l)))" "$OUT" 2>/dev/null || echo 0)
say "mask-factual swap: $N answers without error (248 expected for qwen3-next-80b)"
if [ "${STOP_WHEN_DONE:-0}" = 1 ]; then
  say "=== DONE - stopping the box (STOP_WHEN_DONE=1 set at launch)"
  vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
else
  say "=== ALL STAGES FINISHED - box is IDLE and STILL BILLING; stop it by hand"
fi
