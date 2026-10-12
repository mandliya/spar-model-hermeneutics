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
# Results and the run log go to HF under
# models/<MODEL_TAG>/substitute/gemma-3-27b/boxes/<JOB>/ after every stage and every
# 10 minutes, starting before the model download.
# Nothing here needs a restart of the box: downloads and stages are retried, every
# stage resumes per answer, and pasting the same launch line again continues the job.
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
# One copy per box: pasting the launch line twice must not start a second run on the same GPU.
if command -v flock > /dev/null; then
  exec 9> /workspace/$JOB.lock
  flock -n 9 || { say "!!! $JOB is ALREADY RUNNING on this box - not starting a second copy (check: tail -3 /workspace/job_$JOB.log)"; exit 1; }
fi
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
fail () { say "!!! $* - aborting this job before the full run"; kill $PUSHER 2>/dev/null; push; stop_box; exit 1; }
retry () {   # retry <cmd...>: up to 3 attempts, for steps that only fail on a network hiccup
  local a
  for a in 1 2 3; do
    say "=== $* (attempt $a)"; "$@" && return 0
    say "!!! failed: $*"; sleep 30
  done
  return 1
}
stage () {   # stage <label> <args of 16_substitute_followups.py...>
  # Up to 3 attempts. The script resumes per answer and retries errored rows, so an
  # attempt that crashed or left rows unanswered (exit 2) carries on where it stopped.
  # A failed gate (exit 3, unparseable answers) is never retried.
  local label=$1 a rc; shift
  for a in 1 2 3; do
    say "##### $label (attempt $a)"
    python 16_substitute_followups.py "$@"; rc=$?
    push
    [ $rc = 0 ] && { say "--- $label: complete"; return 0; }
    [ $rc = 3 ] && { say "!!! $label: GATE FAILED - not retried"; return 3; }
    say "!!! $label: stopped early or left answers missing (exit $rc) - resuming in 60 s"; sleep 60
  done
  say "!!! $label: still incomplete after 3 attempts"; return 1
}

# --- setup --------------------------------------------------------------------
say "=== setup: packages"
for attempt in 1 2 3; do
  pip install -q "transformers==5.17.0" accelerate pandas pyarrow fsspec huggingface_hub \
    python-dotenv tqdm sentencepiece protobuf && say "--- packages OK" && break
  say "pip attempt $attempt failed, retrying"; sleep 20
done
python -c 'import transformers, torch, pandas, pyarrow, fsspec, dotenv, tqdm, accelerate; print("transformers", transformers.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())' \
  || fail "packages missing after 3 pip attempts"
push   # the log is on HF from here on, so the box can be checked without opening it
( exec 9>&-; while sleep 600; do push > /dev/null 2>&1; done ) &   # 9>&-: the pusher must not hold the lock
PUSHER=$!
say "=== setup: $READER_ID"
for attempt in 1 2 3 4 5; do
  [ $attempt -ge 2 ] && export HF_HUB_DISABLE_XET=1   # plain HTTP after an xet failure or hang
  # a download that hangs is killed after 40 minutes and retried; finished files are kept
  timeout 2400 python -c "from huggingface_hub import snapshot_download; print('DOWNLOADED', snapshot_download('$READER_ID'))" && break
  say "download attempt $attempt failed or hung, retrying"; sleep 30
done
python -c "from huggingface_hub import snapshot_download; snapshot_download('$READER_ID', local_files_only=True)" \
  || fail "model download failed 5 times - nothing run"

# --- render check: every job of this box, tokenizer only ----------------------
retry python 16_substitute_followups.py --render 0 --categories $CSV || fail "RENDER CHECK FAILED"
retry python 16_substitute_followups.py --render 0 --categories $CSV --swap || fail "RENDER CHECK FAILED (swapped order)"

# --- smoke test: 2 conversations per category, every question, both letter orders.
#     All stages resume, so the full run continues from these rows. ------------
T0=$SECONDS
# Aborts on a failed gate (unparseable answers) or a category with fewer than 6 of its 8
# answers; one row that keeps erroring does not abort the job.
stage "SMOKE TEST, original letter order" --categories $CSV --limit 2 $ACT
[ $? = 3 ] && fail "SMOKE TEST FAILED (gate)"
for c in $CATS; do
  n=$(ok_rows $R/pa_$c.jsonl)
  [ "$n" -ge 6 ] && say "--- smoke test OK: $c ($n of 8 answers)" || fail "SMOKE TEST FAILED: $c ($n of 8 answers)"
done
stage "SMOKE TEST, swapped letter order" --categories $CSV --limit 2 --swap
[ $? = 3 ] && fail "SMOKE TEST FAILED (gate, swapped order)"
say "##### SMOKE TEST PASSED for every category: $CATS ($((SECONDS - T0))s, model loaded twice)"; push
if [ "${SMOKE_ONLY:-0}" = 1 ]; then
  kill $PUSHER 2>/dev/null
  say "=== SMOKE_ONLY: stopping here. Box is IDLE and STILL BILLING unless STOP_WHEN_DONE=1."
  stop_box; exit 0
fi

# --- full run. The two letter orders are separate stages: one failing does not
#     stop the other. ------------------------------------------------------------
stage "FULL RUN, original letter order" --categories $CSV $ACT; RC_ORIG=$?
stage "FULL RUN, swapped letter order" --categories $CSV --swap; RC_SWAP=$?
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
[ "$RC_ORIG" = 0 ] && [ "$RC_SWAP" = 0 ] && say "##### JOB COMPLETE: every answer saved, both letter orders" \
  || say "!!! JOB INCOMPLETE: original order exit $RC_ORIG, swapped order exit $RC_SWAP - paste the launch line again to resume"
if [ "${STOP_WHEN_DONE:-0}" = 1 ]; then   # always stop at the end; the check above is only logged
  say "=== ALL DONE - stopping the box (STOP_WHEN_DONE=1 set at launch)"
  vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
else
  say "=== ALL STAGES FINISHED - box is IDLE and STILL BILLING; stop it by hand"
fi
