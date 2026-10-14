#!/usr/bin/env bash
# One box = one job: the full gemma pipeline for this job's categories, on the
# model named by MODEL_TAG (src/common.py: qwen3-32b, qwen3-next-80b), one
# conversation at a time. MODEL_TAG must be set at launch.
#   setup -> model download -> per category: smoke test (gate), first run,
#   follow-ups, named_cot, activations, P("A") -> A/B swap -> transcripts ->
#   monitors only if MONITORS=1 (the model itself, batched, on the GPU; Gemini +
#   GPT-5 mini via API at the same time) -> activations packaged.
# SMOKE_ONLY=1: stop after the smoke test (timings for a cost quote); job SMOKE
# runs the smoke test on every category.
# Results go to HF under models/<MODEL_TAG>/boxes/<JOB>/ after every stage.
# A category that fails is logged and skipped; the other categories still run.
# The box is NOT stopped unless STOP_WHEN_DONE=1 was set at launch (only on
# the user's say-so), and then only after the uploads are verified.
#
#   bash run_job.sh J1      (jobs listed below)
JOB=$1
cd "$(dirname "$0")"
say () { echo "[$(date -u +%H:%M:%SZ)] [$JOB] $*"; }
source /venv/main/bin/activate 2>/dev/null || true
export HF_HOME=/workspace/.hf_home PYTHONUNBUFFERED=1

case $JOB in
  J1)  CATS="alibi";          export SHARD=0/3 ;;
  J2)  CATS="alibi";          export SHARD=1/3 ;;
  J3)  CATS="alibi";          export SHARD=2/3 ;;
  J4)  CATS="capture_secret"; export SHARD=0/2 ;;
  J5)  CATS="capture_secret"; export SHARD=1/2 ;;
  J6)  CATS="ascii mask_factual" ;;
  J7)  CATS="sycophancy_a sycophancy_b" ;;
  J8)  CATS="unanswerable mask_known_facts mask_statistics" ;;
  J9)  CATS="mask_continuations" ;;
  J10) CATS="mask_disinformation" ;;
  J6b) CATS="mask_factual" ;;   # mask-factual again: J6 stopped on the empty-trace rows
  SMOKE) CATS="ascii sycophancy_a sycophancy_b mask_factual unanswerable capture_secret alibi mask_known_facts mask_statistics mask_continuations mask_disinformation"
         SMOKE_ONLY=1 ;;   # every category's smoke test, then stop: per-category timings
  SMOKE3) CATS="ascii alibi mask_known_facts"
         SMOKE_ONLY=1 ;;   # three categories only (short prompts, long multi-turn, MASK belief): for a model
                           # that takes minutes to load, the full SMOKE would mostly pay for ~90 model loads
  *) echo "unknown job '$JOB' (J1-J10, J6b, SMOKE, SMOKE3)"; exit 1 ;;
esac
# No default model: a launch line without MODEL_TAG would silently rerun the wrong one.
case "${MODEL_TAG:-}" in
  qwen3-32b)      MIN_GPU_MIB=75000;  MIN_DISK_GB=150; MODEL_ID=Qwen/Qwen3-32B; MODEL_GB=60;  DL_TIMEOUT=3600 ;;
  qwen3-next-80b) MIN_GPU_MIB=170000; MIN_DISK_GB=250; MODEL_ID=Qwen/Qwen3-Next-80B-A3B-Instruct; MODEL_GB=150; DL_TIMEOUT=7200 ;;   # 163 GB of weights
  qwen3-235b)     MIN_GPU_MIB=540000; MIN_DISK_GB=620; MODEL_ID=Qwen/Qwen3-235B-A22B-Instruct-2507; MODEL_GB=440; DL_TIMEOUT=14400 ;;   # 470 GB of weights: 8 x 96 GB or 4 x 141 GB
  *) echo "MODEL_TAG must be set at launch (qwen3-32b, qwen3-next-80b or qwen3-235b), got '${MODEL_TAG:-}'"; exit 1 ;;
esac
export MODEL_TAG
# One copy per box: pasting the launch line twice must not start a second run on the same GPUs.
if command -v flock > /dev/null; then
  exec 9> /workspace/$JOB.lock
  flock -n 9 || { say "!!! $JOB is ALREADY RUNNING on this box - not starting a second copy (check: tail -3 /workspace/job_$JOB.log)"; exit 1; }
fi
say "model: $MODEL_TAG | categories: $CATS | shard: ${SHARD:-all rows} | Vast instance: ${CONTAINER_ID:-unknown}"
# Dedicated GPU memory, summed over every GPU (device_map="auto" spreads the model).
# A GB10 (DGX Spark, shared CPU/GPU memory, reported as [N/A]) or too little memory
# makes transformers offload layers to disk and run at a crawl - refuse instead.
GPUS=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
MEM=$(echo "$GPUS" | awk -F', ' '{s += $2} END {print s + 0}')
DISK=$(df -BG --output=avail /workspace 2>/dev/null | tail -1 | awk '{print $NF}' | tr -dc '0-9')
# A second job on the same box finds the model already downloaded: then it needs only
# room for results, not for another copy (9 Oct: J7 and J10 were refused for this and
# their boxes stopped).
CACHED=$(du -sBG "$HF_HOME/hub/models--${MODEL_ID//\//--}" 2>/dev/null | cut -f1 | tr -dc '0-9')
[ "${CACHED:-0}" -ge "$MODEL_GB" ] && MIN_DISK_GB=30
if echo "$GPUS" | grep -qi "GB10" || ! [ "${MEM:-0}" -ge "$MIN_GPU_MIB" ] 2>/dev/null || ! [ "${DISK:-0}" -ge "$MIN_DISK_GB" ] 2>/dev/null; then
  say "!!! WRONG BOX: GPUs '$(echo $GPUS | tr '\n' ';')' = $MEM MiB (need >= $MIN_GPU_MIB), disk ${DISK:-?} GB free (need >= $MIN_DISK_GB). Not running."
  [ "${STOP_WHEN_DONE:-0}" = 1 ] && vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
  exit 1
fi
say "GPU OK: $(echo "$GPUS" | wc -l | tr -d ' ') x $(echo "$GPUS" | head -1 | cut -d, -f1), $MEM MiB total | disk $DISK GB free"
# Two boxes (9 Oct) produced garbage text: the model split over two GPUs, with GPU-to-GPU
# copies silently corrupted on those machines. Copy data between the GPUs and compare
# before downloading anything. Refuses ONLY on a seen mismatch; if the test itself
# errors, the job goes on (the smoke test still catches garbage text).
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
  [ "${STOP_WHEN_DONE:-0}" = 1 ] && vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
  exit 1
fi
[ -n "$HF_TOKEN" ] && [ -n "$OPENROUTER_API_KEY" ] || { say "!!! HF_TOKEN and OPENROUTER_API_KEY must be set"; exit 1; }
printf 'HF_TOKEN=%s\nOPENROUTER_API_KEY=%s\nJUDGE_MODEL=anthropic/claude-sonnet-4.5\n' "$HF_TOKEN" "$OPENROUTER_API_KEY" > .env
chmod 600 .env

# --- helpers ------------------------------------------------------------------
R=results/$MODEL_TAG
push () {   # every results file (not tensors) -> models/<tag>/boxes/<JOB>/results
  python - "$JOB" <<'PY'
import os, sys
from huggingface_hub import HfApi
from src import common as C
from src.hf_sync import REPO, remote
job = sys.argv[1]
for attempt in range(4):
    try:
        HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
            folder_path=str(C.RESULTS), path_in_repo=remote(f"boxes/{job}/results"), repo_id=REPO,
            repo_type="dataset", ignore_patterns=["activations/*", "*.pt"],
            commit_message=f"{C.MODEL_TAG} {job}: results")
        print("PUSHED results", flush=True); break
    except Exception as e:
        print("push failed:", repr(e)[:200], flush=True)
log = f"/workspace/job_{job}.log"   # the run log, so progress is visible from anywhere
if os.path.exists(log):
    try:
        HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj=log, path_in_repo=remote(f"boxes/{job}/job.log"),
                                                       repo_id=REPO, repo_type="dataset", commit_message=f"{job}: log")
    except Exception as e:
        print("log push failed:", repr(e)[:200], flush=True)
PY
}
final_push () {   # the last log lines must reach HF, or a finished box looks "still running" from outside.
  # Never holds up the end of the job for more than 5 minutes.
  push > /dev/null 2>&1 &
  local p=$! i=0
  while kill -0 $p 2>/dev/null && [ $i -lt 300 ]; do sleep 1; i=$((i + 1)); done
  kill $p 2>/dev/null
  return 0
}
ok_rows () {   # ok_rows <jsonl> -> number of rows without an "error" field
  python -c "import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and 'error' not in json.loads(l)))" "$1" 2>/dev/null || echo 0
}
gate () {      # gate <jsonl> <min ok rows> <what>
  n=$(ok_rows "$1"); if [ "$n" -ge "$2" ]; then say "--- smoke test OK: $3 ($n rows)"; return 0; fi
  say "!!! SMOKE TEST FAILED: $3 ($n of $2 rows) - skipping this category"; return 1
}
run () { say "=== $*"; "$@" || { say "!!! failed: $*"; return 1; }; }

# --- setup --------------------------------------------------------------------
say "=== setup: packages"
pip install -q "transformers==5.17.0" accelerate pandas pyarrow fsspec huggingface_hub \
  python-dotenv openai tqdm statsmodels requests && say "--- packages OK"
# Qwen3-Next's linear-attention layers: fast kernels from flash-linear-attention
# (Python + Triton, nothing to compile); without it transformers falls back to a
# slow plain-PyTorch path. Pinned so every box computes the same way.
[ "$MODEL_TAG" = qwen3-next-80b ] && pip install -q "flash-linear-attention==0.5.2" && say "--- flash-linear-attention OK"
python -c 'import transformers, torch; print("transformers", transformers.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())'
MODEL=$(python -c "from src import common as C; print(C.TARGET_MODEL)")
push   # the log is on HF from here on, so the box can be checked without opening it
( exec 9>&-; while sleep 600; do push > /dev/null 2>&1; done ) &   # ...and every 10 minutes; 9>&-: must not hold the lock
PUSHER=$!
say "=== setup: $MODEL"
for attempt in 1 2 3 4 5; do
  # from attempt 2: plain HTTP instead of HF's xet transfer (J6 hit xet "CAS Client
  # Error ... error decoding response body" three times in a row)
  [ $attempt -ge 2 ] && export HF_HUB_DISABLE_XET=1
  # a download that hangs is killed after DL_TIMEOUT and retried; finished files are kept
  timeout $DL_TIMEOUT python -c "from huggingface_hub import snapshot_download; print('DOWNLOADED', snapshot_download('$MODEL'))" && break
  say "download attempt $attempt failed or hung, retrying"; sleep 30
done
say "--- model on disk: $(du -sh "$HF_HOME/hub/models--${MODEL_ID//\//--}" 2>/dev/null | cut -f1)"
if ! python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', local_files_only=True)"; then
  say "!!! model download failed 5 times - nothing run"
  kill $PUSHER 2>/dev/null; final_push
  [ "${STOP_WHEN_DONE:-0}" = 1 ] && { say "stopping the box (STOP_WHEN_DONE=1)"; vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"; }
  exit 1
fi

# --- one category, start to finish ------------------------------------------
# chain <category> <L>: every stage for one category. L="" = all rows; L=2 =
# the smoke test (2 rows/jobs through every stage). All stages resume, so the
# full run continues from the smoke rows.
followup_cats=""   # categories with the standard follow-ups (for swap + transcripts)
chain () {
  local c=$1 L=$2 lim n fl
  lim=${L:+--limit $L}; n=${L:+--n $L}
  fl=${L:+--limit $((L * 4))}   # follow-up jobs are row x question: cover every question for the smoke rows
  case $c in
    alibi)          run python 00_alibi.py $lim || return 1; FIRST=alibi_conditions.jsonl
                    run python 00_alibi_selfreport_benchmark.py $fl || return 1 ;;
    capture_secret) # smoke: 3x the rows - follow-ups only use rows where the secret was pressed (~75%)
                    run python 00_capture_secret.py ${L:+--limit $((L * 3))} || return 1; FIRST=capture_secret_conditions.jsonl
                    run python 00_capture_secret_selfreport_benchmark.py $fl || return 1 ;;
    ascii)          run python 00_ascii_activations.py $n || return 1; FIRST=ascii_results.jsonl
                    run python 00_ascii_reasoning.py $fl || return 1
                    run python 00_ascii_selfreport_benchmark.py $fl || return 1 ;;
    sycophancy_a)   run python 00_sycophancy_answers.py $lim || return 1; FIRST=sycophancy_answer_conditions.jsonl
                    run python 00_sycophancy_selfreport_benchmark.py $fl || return 1 ;;
    sycophancy_b)   run python 00_sycophancy_feedback.py $lim || return 1; FIRST=sycophancy_feedback_conditions.jsonl
                    run python 00_sycophancy_feedback_selfreport_benchmark.py $fl || return 1 ;;
    mask_factual)   run python 00_mask_factual_provided.py $lim || return 1; FIRST=mask_factual_provided_conditions.jsonl
                    run python 00_mask_factual_selfreport_benchmark.py $fl || return 1 ;;
    unanswerable)   run python 00_unanswerable_claims.py $lim || return 1; FIRST=unanswerable_claims.jsonl
                    run python 00_unanswerable_selfreport_benchmark.py $fl || return 1 ;;
    mask_*)         local sub=${c#mask_}
                    run python 12_mask_belief_generate.py --subtypes $sub $lim || return 1; FIRST=mask_belief_${sub}_answers.jsonl
                    run python 13_mask_belief_judge.py --subtypes $sub $lim || return 1
                    run python 14_mask_belief_followups.py --subtypes $sub $fl || return 1
                    run python 02_analyse_mask_belief.py || return 1
                    return 0 ;;   # MASK: the four questions + P("A") come from 14; no swap test (as for gemma)
  esac
  run python 05_named_cot.py --category $c $lim || return 1
  case $c in ascii|sycophancy_a|sycophancy_b|mask_factual) run python 04_recapture_fp32.py --category $c $lim || return 1 ;; esac
  run python 06_answer_probability.py --categories $c ${L:+--limit 5} || return 1
}

# smoke test: every category, every stage, 2 rows. Any failure aborts the job
# before the long run (box left running, unless STOP_WHEN_DONE=1).
TIMES=""
for c in $CATS; do
  say "##### SMOKE TEST: $c"
  T0=$SECONDS
  if chain $c 2 && gate $R/$FIRST 2 "$c first run"; then TIMES="$TIMES $c=$((SECONDS - T0))s"; else
    say "!!! SMOKE TEST FAILED for $c - aborting this job before the full run"; kill $PUSHER 2>/dev/null; final_push
    [ "${STOP_WHEN_DONE:-0}" = 1 ] && vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
    exit 1
  fi
done
say "##### SMOKE TEST PASSED for every category: $CATS"
say "smoke-test seconds per category (2 rows, every stage):$TIMES"; push
if [ "${SMOKE_ONLY:-0}" = 1 ]; then
  say "=== SMOKE_ONLY: stopping here. Box is IDLE and STILL BILLING unless STOP_WHEN_DONE=1."
  kill $PUSHER 2>/dev/null; final_push
  [ "${STOP_WHEN_DONE:-0}" = 1 ] && vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
  exit 0
fi

for c in $CATS; do
  say "##### FULL RUN: $c"
  if chain $c ""; then
    # only the four MASK belief categories have no swap test. This was `mask_*`, which
    # also matched mask_factual: neither Qwen run got a mask-factual swap test.
    case $c in mask_known_facts|mask_statistics|mask_continuations|mask_disinformation) ;; *) followup_cats="$followup_cats $c" ;; esac
    say "--- $c done"
  else
    say "!!! $c stopped early (see above); continuing with the rest"
  fi
  push
done

# --- A/B swap on this box's categories (MASK excluded, as for gemma) ----------
if [ -n "$followup_cats" ]; then
  run python 00_pick_subset.py && run python 07_ab_swap.py --categories $(echo $followup_cats | tr ' ' ',')
  push
fi

# --- transcripts + monitors: Gemini/GPT via API (closed models) in the
#     background while the model reads its own transcripts on the GPU ---------
# (MONITORS=1 only: off by default since the Qwen3-32B run; transcripts are always built)
if run python 08_build_transcripts.py --categories $(echo $CATS | tr ' ' ',') && [ "${MONITORS:-0}" = 1 ]; then
  python 10_monitor_api.py --models all > /workspace/monitor_api_$JOB.log 2>&1 &
  API=$!
  run python 09_monitor_self.py
  wait $API && say "--- API monitors done: $(tail -1 /workspace/monitor_api_$JOB.log)" \
    || say "!!! API monitors failed (resumable): $(tail -2 /workspace/monitor_api_$JOB.log)"
fi
push

# --- activations: one tarball per job ----------------------------------------
say "=== packaging activations"
TB=/workspace/activations_$JOB.tar.gz
DIRS=""
[ -d $R/activations ] && DIRS="$DIRS $PWD/$R/activations"
[ -d /workspace/fp32_staging ] && DIRS="$DIRS /workspace/fp32_staging"
[ -d /workspace/mask_belief_staging ] && DIRS="$DIRS /workspace/mask_belief_staging"
NPT=$(find $DIRS -name "*.pt" 2>/dev/null | wc -l)
say "$NPT tensors"
if [ "$NPT" -gt 0 ] && tar -czf $TB $DIRS 2>/dev/null; then
  python -c "
from src.hf_sync import upload
import sys
sys.exit(0 if upload('$TB', 'boxes/$JOB/activations_$JOB.tar.gz') else 1)" && say "--- activations uploaded ($(du -h $TB | cut -f1))" \
    || say "!!! activations upload failed - tarball kept at $TB"
fi

# --- verify, then stop only if the user asked for it at launch -----------------
VERIFIED=$(python - "$JOB" "$TB" <<'PY' | tail -1
import os, sys, pathlib
from huggingface_hub import HfApi
from src import common as C
from src.hf_sync import REPO, remote
job, tb = sys.argv[1], pathlib.Path(sys.argv[2])
api = HfApi(token=os.environ["HF_TOKEN"])
ok = True
for f in C.RESULTS.glob("*.jsonl"):
    info = api.get_paths_info(REPO, [remote(f"boxes/{job}/results/{f.name}")], repo_type="dataset")
    ok &= bool(info) and info[0].size == f.stat().st_size
if tb.exists():
    info = api.get_paths_info(REPO, [remote(f"boxes/{job}/{tb.name}")], repo_type="dataset")
    ok &= bool(info) and info[0].size == tb.stat().st_size
print("yes" if ok else "no")
PY
)
say "results verified on HF: $VERIFIED"
kill $PUSHER 2>/dev/null
if [ "${STOP_WHEN_DONE:-0}" = 1 ]; then   # always stop at the end; the check above is only logged
  say "=== ALL DONE and verified - stopping the box (STOP_WHEN_DONE=1 set at launch)"
  final_push
  vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
else
  say "=== ALL STAGES FINISHED - box is IDLE and STILL BILLING; stop it by hand"
  final_push
fi
