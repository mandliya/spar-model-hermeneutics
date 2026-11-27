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
  # Qwen3.6 with vLLM text (26 Nov 2026): own job names, so their HF folders don't mix with the transformers SMOKE3
  VSMOKE3) CATS="ascii alibi mask_known_facts"
         SMOKE_ONLY=1 ;;   # SMOKE3 through the two passes (text from vLLM, activations from transformers)
  VALL) CATS="ascii sycophancy_a sycophancy_b mask_factual unanswerable capture_secret alibi mask_known_facts mask_statistics mask_continuations mask_disinformation" ;;
  VA)   CATS="ascii alibi mask_known_facts" ;;   # two Qwen3.6 boxes, VA + VB = VALL: VA continues on the VSMOKE3 box
  VASCII) CATS="ascii" ;;   # VA's ascii again (26 Nov): the first run's summary crashed on a None, so its later stages never ran
  VBFIX) CATS="mask_factual sycophancy_a sycophancy_b unanswerable" ;;   # VB again (27 Nov): mask_factual's P("A") gate and
                           # the text pass of 07_ab_swap (crashed after its first category) failed; every stage resumes
  VB)   CATS="capture_secret sycophancy_a sycophancy_b mask_factual unanswerable mask_statistics mask_continuations mask_disinformation" ;;
  *) echo "unknown job '$JOB' (J1-J10, J6b, SMOKE, SMOKE3, VSMOKE3, VALL, VA, VASCII, VB, VBFIX)"; exit 1 ;;
esac
# No default model: a launch line without MODEL_TAG would silently rerun the wrong one.
case "${MODEL_TAG:-}" in
  qwen3-32b)      MIN_GPU_MIB=75000;  MIN_DISK_GB=150; MODEL_ID=Qwen/Qwen3-32B; MODEL_GB=60;  DL_TIMEOUT=3600 ;;
  qwen3-next-80b) MIN_GPU_MIB=170000; MIN_DISK_GB=250; MODEL_ID=Qwen/Qwen3-Next-80B-A3B-Instruct; MODEL_GB=150; DL_TIMEOUT=7200 ;;   # 163 GB of weights
  qwen3-235b)     MIN_GPU_MIB=540000; MIN_DISK_GB=620; MODEL_ID=Qwen/Qwen3-235B-A22B-Instruct-2507; MODEL_GB=420; DL_TIMEOUT=14400 ;;   # 470 GB of weights (438 GiB as du counts it): 8 x 96 GB or 4 x 141 GB
  # 26 Nov 2026 (TQ): thinking ON, reasoning kept in every row (src/common.py, THINKING); ~54 GB of weights
  qwen3.6-27b)    MIN_GPU_MIB=75000;  MIN_DISK_GB=120; MODEL_ID=Qwen/Qwen3.6-27B; MODEL_GB=45;  DL_TIMEOUT=3600; export THINKING=1
                  export HF_HUB_DISABLE_XET=1      # xet hung or restarted files on both smoke boxes (26 Nov); plain HTTP ran at ~68 MB/s
                  export VLLM_GEN=1 ;;             # text from vLLM, activations from transformers (GG, 26 Nov; see run () below)
  *) echo "MODEL_TAG must be set at launch (qwen3-32b, qwen3-next-80b, qwen3-235b or qwen3.6-27b), got '${MODEL_TAG:-}'"; exit 1 ;;
esac
export MODEL_TAG
[ "${THINKING:-}" = 1 ] && say "thinking mode ON: +${THINK_BUDGET:-8192} new tokens per generation, model-card sampling seeded per prompt"
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
# du counts GiB: 470 GB of weights show as 438G, so MODEL_GB must sit below the GiB figure
# (14 Oct: MODEL_GB=440 refused three jobs on the box that already had the 235B).
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
run () {
  say "=== $*"
  # VLLM_GEN=1 (Qwen3.6, GG's choice 26 Nov 2026): a script that generates text runs twice. The text pass: GEN_P copies
  # at once (GEN_PASS=text, each with every GEN_P-th row) send their prompts to a local vLLM server, which batches them,
  # and the answers land in /workspace/gen_cache/<script>/. The real pass (GEN_PASS=real): the unchanged script with
  # the transformers model, which takes each answer from the cache and computes activations, P("A") and the rows as
  # before. vLLM is stopped between the two, so the real pass has the whole GPU. Details: src/common.py, GEN_PASS.
  if [ "${VLLM_GEN:-0}" = 1 ] && [ "$1" = python ] && [[ " $GEN_SCRIPTS " == *" $2 "* ]]; then
    text_pass "${@:2}" || { say "!!! text pass failed: $*"; return 1; }
    # the real pass needs the whole GPU: a model loaded next to a leftover server offloads to the CPU or runs out of memory
    [ "$(gpu_used_mib)" -lt 2000 ] || { say "!!! GPU still in use ($(gpu_used_mib) MiB) after stopping vLLM: not starting the real pass of $2"; return 1; }
    local m0 m1; m0=$(cat "/workspace/gen_cache/${2%.py}/misses.log" 2>/dev/null | wc -l)
    GEN_PASS=real "$@" || { say "!!! failed (real pass): $*"; return 1; }
    m1=$(cat "/workspace/gen_cache/${2%.py}/misses.log" 2>/dev/null | wc -l)
    say "--- real pass done: $((m1 - m0)) prompts were not in the vLLM cache (generated by transformers)"
    return 0
  fi
  "$@" || { say "!!! failed: $*"; return 1; }
}
GEN_SCRIPTS="00_alibi.py 00_alibi_selfreport_benchmark.py 00_ascii_activations.py 00_ascii_reasoning.py
  00_ascii_selfreport_benchmark.py 00_capture_secret.py 00_capture_secret_selfreport_benchmark.py
  00_mask_factual_provided.py 00_mask_factual_selfreport_benchmark.py 00_sycophancy_answers.py 00_sycophancy_feedback.py
  00_sycophancy_feedback_selfreport_benchmark.py 00_sycophancy_selfreport_benchmark.py 00_unanswerable_claims.py
  00_unanswerable_selfreport_benchmark.py 05_named_cot.py 07_ab_swap.py 12_mask_belief_generate.py 14_mask_belief_followups.py"
GEN_SCRIPTS=$(echo $GEN_SCRIPTS)   # one line, single spaces
GEN_P=${GEN_P:-48}                 # text-pass copies at once (vLLM batches their requests)
vllm_up () {
  curl -sf -m 5 http://127.0.0.1:8000/health > /dev/null && return 0
  say "--- starting vLLM"
  # VLLM_USE_FLASHINFER_SAMPLER=0: FlashInfer compiles its sampling kernel at startup, and the image's nvcc 12.1 rejects
  # its flags (26 Nov); PyTorch's top-k/top-p sampler instead (same settings, a little slower)
  ( cd /workspace && HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 setsid nohup /venv/vllm/bin/vllm serve "$MODEL_ID" --served-model-name qwen \
      --max-model-len 32768 --gpu-memory-utilization 0.92 --enable-prefix-caching --max-num-seqs 64 --port 8000 \
      > /workspace/vllm_server.log 2>&1 < /dev/null & )
  local i; for i in $(seq 1 120); do
    sleep 5; curl -sf -m 5 http://127.0.0.1:8000/health > /dev/null && { say "--- vLLM up after $((i * 5)) s"; return 0; }
    [ $i -ge 6 ] && [ -z "$(vllm_groups)" ] && break   # the server process is gone: it failed
  done
  say "!!! vLLM did not start: $(tail -n 3 /workspace/vllm_server.log | tr '\n' ' ' | cut -c1-400)"; vllm_down; return 1
}
# The vLLM server's process groups, found by command line ($! is a bash wrapper, not the server: on 26 Nov the
# server outlived every "stop", and the real passes loaded transformers next to it). setsid gives the server its own
# group; never this script's own group.
vllm_groups () {
  local me; me=$(ps -o pgid= -p $$ | tr -d ' ')
  ps -eo pgid=,args= | awk -v me="$me" '$2 == "/venv/vllm/bin/python" && $3 == "/venv/vllm/bin/vllm" && $4 == "serve" && $1 != me && $1 > 1 {print $1}' | sort -u
}
gpu_used_mib () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -n 1; }
vllm_down () {
  local g i
  for g in $(vllm_groups); do kill -- -"$g" 2> /dev/null; done
  for i in $(seq 1 60); do [ "$(gpu_used_mib)" -lt 2000 ] && return 0; sleep 3; done
  for g in $(vllm_groups); do kill -9 -- -"$g" 2> /dev/null; done
  sleep 10; [ "$(gpu_used_mib)" -lt 2000 ]
}
text_pass () {   # text_pass <script> <args...>
  vllm_up || return 1
  rm -rf /workspace/gen_scratch && mkdir -p /workspace/gen_scratch/results /workspace/gen_logs
  rsync -a --exclude activations/ --exclude archive/ "$R/" /workspace/gen_scratch/results/ || return 1
  local k pids="" bad=0 n0 n1 name=${1%.py}
  n0=$(cat /workspace/gen_cache/$name/part*.jsonl 2> /dev/null | wc -l)
  for k in $(seq 0 $((GEN_P - 1))); do
    # small thread pools: by default each copy sizes its tokenizer (Rust) and torch (OpenMP) pools to all 256 cores,
    # and 48 copies passed the container's 7,680-thread cap (pids.max): 9 of 48 crashed at start (26 Nov)
    # HF_HUB_OFFLINE=1: everything a copy reads from the Hub is already on disk (model files, the two datasets,
    # fetched in setup); 48 copies asking the Hub at once hit its 1,000-calls-per-5-minutes limit (26 Nov)
    CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 GEN_PASS=text GEN_PART=$k/$GEN_P RESULTS_DIR=/workspace/gen_scratch/results \
      RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python "$@" > /workspace/gen_logs/${name}_$k.log 2>&1 &
    pids="$pids $!"
  done
  for k in $pids; do wait $k || bad=$((bad + 1)); done
  n1=$(cat /workspace/gen_cache/$name/part*.jsonl 2> /dev/null | wc -l)
  # a copy may exit non-zero AFTER its loop (summaries read rows the text pass never writes); the real pass's
  # cache-miss count is the check that matters
  vllm_down || { say "!!! vLLM did not stop"; return 1; }
  say "--- text pass $1: $((n1 - n0)) answers from vLLM, $GEN_P copies ($bad exited non-zero; logs /workspace/gen_logs/${name}_*.log)"
  grep -hE "vLLM request failed|Error|Panic" /workspace/gen_logs/${name}_*.log 2> /dev/null | grep -v "^FileNotFoundError: .*gen_scratch/results/" \
    | sort | uniq -c | sort -rn | head -n 3 | cut -c1-200 | while read -r l; do say "    $l"; done
}

# --- setup --------------------------------------------------------------------
say "=== setup: packages"
pip install -q "transformers==5.17.0" accelerate pandas pyarrow fsspec huggingface_hub \
  python-dotenv openai tqdm statsmodels requests && say "--- packages OK"
# Qwen3-Next's linear-attention layers: fast kernels from flash-linear-attention
# (Python + Triton, nothing to compile); without it transformers falls back to a
# slow plain-PyTorch path. Pinned so every box computes the same way.
# Qwen3.6 (26 Nov 2026): the vastai/pytorch image can ship torch 2.5.1 with Triton 3.1, too old for fla 0.5.2: the
# package installs but `import fla` fails, and transformers silently uses the slow path (SMOKE3 ran that way). So
# Qwen3.6 boxes pin torch 2.7.1 (Triton 3.3.1) and causal-conv1d, and the job stops if either kernel does not import.
case $MODEL_TAG in qwen3.6-27b)
  pip install -q "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" --index-url https://download.pytorch.org/whl/cu126 \
    && say "--- torch 2.7.1 OK" ;; esac
case $MODEL_TAG in qwen3-next-80b|qwen3.6-27b)   # Qwen3.6 has the same Gated DeltaNet layers (26 Nov 2026)
  pip install -q "flash-linear-attention==0.5.2" && say "--- flash-linear-attention OK" ;; esac
case $MODEL_TAG in qwen3.6-27b)   # vLLM in its own venv: it brings its own torch; the transformers venv stays as pinned
  [ -x /venv/vllm/bin/vllm ] || { python -m pip install -q uv && uv venv /venv/vllm --python 3.12 -q \
    && VIRTUAL_ENV=/venv/vllm uv pip install -q "vllm==0.32.0"; }
  /venv/vllm/bin/python -c "import vllm" 2> /dev/null && say "--- vLLM 0.32.0 OK" || { say "!!! vLLM install failed. Stopping."; exit 1; } ;; esac
case $MODEL_TAG in qwen3.6-27b)
  pip install -q --no-build-isolation "causal-conv1d==1.7.0"
  python -c "import fla, causal_conv1d" > /dev/null 2>&1 && say "--- fast kernels import (fla, causal_conv1d)" \
    || { say "!!! fla or causal_conv1d does not import: transformers would use the slow path. Stopping."; exit 1; } ;; esac
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

# --- the source datasets, on disk once per box (26 Nov 2026): every script then reads them locally (C.load_config,
#     src/mask_belief.py) and the vLLM text-pass copies can run offline
if python - <<'PY'
from huggingface_hub import snapshot_download
from src import common as C
print("lies:", C._hub_retry(lambda: snapshot_download(C.HF_DATASET, repo_type="dataset", revision=C.HF_DATASET_REVISION,
                                                      allow_patterns=["*.parquet"])))
print("MASK:", C._hub_retry(lambda: snapshot_download("cais/MASK", repo_type="dataset",
                                                      allow_patterns=["*/test-00000-of-00001.parquet"])))
PY
then say "--- source datasets on disk"
else say "!!! could not fetch the source datasets"; kill $PUSHER 2>/dev/null; final_push; exit 1
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
