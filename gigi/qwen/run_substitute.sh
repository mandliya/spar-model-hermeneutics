#!/usr/bin/env bash
# One box = one job of the substitute step: gemma-3-27b answers the four follow-up
# questions on the TARGET model's conversations (16_substitute_followups.py).
# MODEL_TAG names the target (qwen3-next-80b) and must be set at launch; its
# results/<MODEL_TAG>/ folder comes with the code bundle and is only read.
#   box check -> packages -> gemma download -> render check (gate, no GPU) ->
#   smoke test: 2 conversations per category, both letter orders (gate) ->
#   full run -> swapped letter order -> activations packaged.
# No API calls: only HF_TOKEN is needed.
# ACTIVATIONS=0: don't save activations (default 1: both capture points, float32).
# SMOKE_ONLY=1, or job SMOKE: stop after the smoke test (timings for a cost quote).
# Results go to HF under models/<MODEL_TAG>/substitute/gemma-3-27b/boxes/<JOB>/
# after every stage and every 20 minutes.
# The box is NOT stopped unless STOP_WHEN_DONE=1 was set at launch (only on the
# user's say-so).
#
#   MODEL_TAG=qwen3-next-80b bash run_substitute.sh S1      (jobs listed below)
JOB=$1
cd "$(dirname "$0")"
say () { echo "[$(date -u +%H:%M:%SZ)] [$JOB] $*"; }
source /venv/main/bin/activate 2>/dev/null || true
export HF_HOME=/workspace/.hf_home PYTHONUNBUFFERED=1
export READER_TAG=gemma-3-27b
READER_ID=google/gemma-3-27b-it

ALL="mask_factual unanswerable sycophancy_a sycophancy_b ascii capture_secret alibi mask_known_facts mask_statistics mask_continuations mask_disinformation"
case $JOB in
  S1)  CATS="alibi"; export SHARD=0/2 ;;
  S2)  CATS="alibi"; export SHARD=1/2 ;;
  S3)  CATS="capture_secret ascii" ;;
  S4)  CATS="mask_factual unanswerable sycophancy_a sycophancy_b mask_known_facts mask_statistics mask_continuations mask_disinformation" ;;
  ALL) CATS=$ALL ;;                    # everything on one box
  SMOKE) CATS=$ALL; SMOKE_ONLY=1 ;;    # every category's smoke test, then stop
  *) echo "unknown job '$JOB' (S1-S4, ALL, SMOKE)"; exit 1 ;;
esac
# No default target: a launch line without MODEL_TAG would read the wrong model's conversations.
[ "${MODEL_TAG:-}" = qwen3-next-80b ] || { echo "MODEL_TAG must be set at launch (qwen3-next-80b), got '${MODEL_TAG:-}'"; exit 1; }
export MODEL_TAG
R=results/${READER_TAG}_reads_$MODEL_TAG
CSV=$(echo $CATS | tr ' ' ',')
ACT=""; [ "${ACTIVATIONS:-1}" = 1 ] && ACT="--activations"
stop_box () { [ "${STOP_WHEN_DONE:-0}" = 1 ] && { say "stopping the box (STOP_WHEN_DONE=1)"; vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"; }; }
say "reader: $READER_TAG | target: $MODEL_TAG | categories: $CATS | shard: ${SHARD:-all rows} | activations: ${ACT:-off} | Vast instance: ${CONTAINER_ID:-unknown}"

# gemma-3-27b is 55 GB in bf16 and fits on one >= 80 GB GPU. Pinned to GPU 0, so the
# model is never split across GPUs (9 Oct: two boxes corrupted GPU-to-GPU copies). A
# GB10 (DGX Spark, shared memory) or a smaller GPU offloads to disk and crawls - refuse.
GPU0=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
MEM=$(echo "$GPU0" | awk -F', ' '{print $2 + 0}')
DISK=$(df -BG --output=avail /workspace 2>/dev/null | tail -1 | awk '{print $NF}' | tr -dc '0-9')
if echo "$GPU0" | grep -qi "GB10" || ! [ "${MEM:-0}" -ge 75000 ] 2>/dev/null || ! [ "${DISK:-0}" -ge 150 ] 2>/dev/null; then
  say "!!! WRONG BOX: GPU 0 '$GPU0' = ${MEM:-?} MiB (need >= 75000), disk ${DISK:-?} GB free (need >= 150). Not running."
  stop_box; exit 1
fi
export CUDA_VISIBLE_DEVICES=0
say "GPU OK: $GPU0 MiB | disk $DISK GB free"
[ -n "$HF_TOKEN" ] || { say "!!! HF_TOKEN must be set"; exit 1; }
printf 'HF_TOKEN=%s\n' "$HF_TOKEN" > .env
chmod 600 .env

# --- helpers ------------------------------------------------------------------
push () {   # every results file (not tensors) + the run log -> .../boxes/<JOB>/
  python - "$JOB" "$R" <<'PY'
import os, sys
from huggingface_hub import HfApi
from src.hf_sync import REPO, remote
job, folder = sys.argv[1], sys.argv[2]
dest = remote(f"substitute/{os.environ['READER_TAG']}/boxes/{job}")
api = HfApi(token=os.environ["HF_TOKEN"])
if os.path.isdir(folder):
    for attempt in range(4):
        try:
            api.upload_folder(folder_path=folder, path_in_repo=f"{dest}/results", repo_id=REPO,
                              repo_type="dataset", ignore_patterns=["*.pt"],
                              commit_message=f"substitute {job}: results")
            print("PUSHED results", flush=True); break
        except Exception as e:
            print("push failed:", repr(e)[:200], flush=True)
log = f"/workspace/job_{job}.log"   # the run log, so progress is visible from anywhere
if os.path.exists(log):
    try:
        api.upload_file(path_or_fileobj=log, path_in_repo=f"{dest}/job.log", repo_id=REPO,
                        repo_type="dataset", commit_message=f"substitute {job}: log")
    except Exception as e:
        print("log push failed:", repr(e)[:200], flush=True)
PY
}
ok_rows () {   # ok_rows <jsonl> -> number of rows without an "error" field
  python -c "import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and 'error' not in json.loads(l)))" "$1" 2>/dev/null || echo 0
}
run () { say "=== $*"; "$@" || { say "!!! failed: $*"; return 1; }; }
fail () { say "!!! $* - aborting this job before the full run"; push; stop_box; exit 1; }

# --- setup --------------------------------------------------------------------
say "=== setup: packages"
pip install -q "transformers==5.17.0" accelerate pandas pyarrow fsspec huggingface_hub \
  python-dotenv tqdm sentencepiece protobuf && say "--- packages OK"
python -c 'import transformers, torch; print("transformers", transformers.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())'
say "=== setup: $READER_ID"
for attempt in 1 2 3 4 5; do
  [ $attempt -ge 2 ] && export HF_HUB_DISABLE_XET=1   # plain HTTP after an xet failure
  python -c "from huggingface_hub import snapshot_download; print('DOWNLOADED', snapshot_download('$READER_ID'))" && break
  say "download attempt $attempt failed, retrying"; sleep 30
done
python -c "from huggingface_hub import snapshot_download; snapshot_download('$READER_ID', local_files_only=True)" \
  || fail "model download failed 5 times - nothing run"

# --- render check: every job of this box, tokenizer only ----------------------
run python 16_substitute_followups.py --render 0 --categories $CSV || fail "RENDER CHECK FAILED"
run python 16_substitute_followups.py --render 0 --categories $CSV --swap || fail "RENDER CHECK FAILED (swapped order)"

# --- smoke test: 2 conversations per category, every question, both letter orders.
#     All stages resume, so the full run continues from these rows. ------------
T0=$SECONDS
run python 16_substitute_followups.py --categories $CSV --limit 2 $ACT || fail "SMOKE TEST FAILED"
for c in $CATS; do
  n=$(ok_rows $R/pa_$c.jsonl)
  [ "$n" -ge 6 ] && say "--- smoke test OK: $c ($n of 8 answers)" || fail "SMOKE TEST FAILED: $c ($n of 8 answers)"
done
run python 16_substitute_followups.py --categories $CSV --limit 2 --swap || fail "SMOKE TEST FAILED (swapped order)"
say "##### SMOKE TEST PASSED for every category: $CATS ($((SECONDS - T0))s, model loaded twice)"; push
if [ "${SMOKE_ONLY:-0}" = 1 ]; then
  say "=== SMOKE_ONLY: stopping here. Box is IDLE and STILL BILLING unless STOP_WHEN_DONE=1."
  stop_box; exit 0
fi

# --- full run. Results are pushed every 20 minutes while it runs. The two letter
#     orders are separate stages: one failing does not stop the other. ----------
( while sleep 1200; do push > /dev/null 2>&1; done ) &
PUSHER=$!
say "##### FULL RUN: original letter order"
run python 16_substitute_followups.py --categories $CSV $ACT; push
say "##### FULL RUN: swapped letter order"
run python 16_substitute_followups.py --categories $CSV --swap; push
kill $PUSHER 2>/dev/null

# --- activations: one tarball per job ----------------------------------------
TB=/workspace/substitute_activations_$JOB.tar.gz
if [ -n "$ACT" ]; then
  say "=== packaging activations: $(find /workspace/substitute_staging -name '*.pt' 2>/dev/null | wc -l) tensors"
  if tar -czf $TB -C /workspace substitute_staging 2>/dev/null; then
    python -c "
from src.hf_sync import upload
import sys
sys.exit(0 if upload('$TB', 'substitute/$READER_TAG/boxes/$JOB/substitute_activations_$JOB.tar.gz') else 1)" \
      && say "--- activations uploaded ($(du -h $TB | cut -f1))" \
      || say "!!! activations upload failed - tarball kept at $TB"
  else
    say "!!! activations tarball failed - tensors kept in /workspace/substitute_staging"
  fi
fi

# --- verify, then stop only if the user asked for it at launch -----------------
VERIFIED=$(python - "$JOB" "$R" "$TB" <<'PY' | tail -1
import os, sys, pathlib
from huggingface_hub import HfApi
from src.hf_sync import REPO, remote
job, folder, tb = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
dest = remote(f"substitute/{os.environ['READER_TAG']}/boxes/{job}")
api = HfApi(token=os.environ["HF_TOKEN"])
ok = True
for f in folder.glob("*.jsonl"):
    info = api.get_paths_info(REPO, [f"{dest}/results/{f.name}"], repo_type="dataset")
    ok &= bool(info) and info[0].size == f.stat().st_size
if tb.exists():
    info = api.get_paths_info(REPO, [f"{dest}/{tb.name}"], repo_type="dataset")
    ok &= bool(info) and info[0].size == tb.stat().st_size
print("yes" if ok else "no")
PY
)
say "results verified on HF: $VERIFIED"
if [ "${STOP_WHEN_DONE:-0}" = 1 ]; then   # always stop at the end; the check above is only logged
  say "=== ALL DONE - stopping the box (STOP_WHEN_DONE=1 set at launch)"
  vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
else
  say "=== ALL STAGES FINISHED - box is IDLE and STILL BILLING; stop it by hand"
fi
