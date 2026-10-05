#!/usr/bin/env bash
# One box = one job: the full gemma pipeline for this job's categories, on the
# model in src/common.py (Qwen3-32B), one conversation at a time.
#   setup -> model download -> per category: smoke test (gate), first run,
#   follow-ups, named_cot, activations, P("A") -> A/B swap -> transcripts ->
#   monitors (the model itself, batched, on the GPU; Gemini + GPT-5 mini via
#   API at the same time) -> activations packaged.
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
  *) echo "unknown job '$JOB' (J1-J10)"; exit 1 ;;
esac
say "categories: $CATS | shard: ${SHARD:-all rows}"
[ -n "$HF_TOKEN" ] && [ -n "$OPENROUTER_API_KEY" ] || { say "!!! HF_TOKEN and OPENROUTER_API_KEY must be set"; exit 1; }
printf 'HF_TOKEN=%s\nOPENROUTER_API_KEY=%s\nJUDGE_MODEL=anthropic/claude-sonnet-4.5\n' "$HF_TOKEN" "$OPENROUTER_API_KEY" > .env
chmod 600 .env

# --- helpers ------------------------------------------------------------------
R=results/$(python -c "import os; print(os.getenv('MODEL_TAG', 'qwen3-32b'))" 2>/dev/null || echo qwen3-32b)
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
python -c 'import transformers, torch; print("transformers", transformers.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())'
MODEL=$(python -c "from src import common as C; print(C.TARGET_MODEL)")
say "=== setup: $MODEL"
for attempt in 1 2 3; do
  python -c "from huggingface_hub import snapshot_download; print('DOWNLOADED', snapshot_download('$MODEL'))" && break
  say "download attempt $attempt failed, retrying"; sleep 20
done
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', local_files_only=True)" \
  || { say "!!! model download failed 3 times - nothing run (box left running)"; exit 1; }

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
for c in $CATS; do
  say "##### SMOKE TEST: $c"
  if chain $c 2 && gate $R/$FIRST 2 "$c first run"; then :; else
    say "!!! SMOKE TEST FAILED for $c - aborting this job before the full run"; push
    [ "${STOP_WHEN_DONE:-0}" = 1 ] && vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
    exit 1
  fi
done
say "##### SMOKE TEST PASSED for every category: $CATS"; push

for c in $CATS; do
  say "##### FULL RUN: $c"
  if chain $c ""; then
    case $c in mask_*) ;; *) followup_cats="$followup_cats $c" ;; esac
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
if run python 08_build_transcripts.py --categories $(echo $CATS | tr ' ' ','); then
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
VERIFIED=$(python - "$JOB" "$TB" <<'PY'
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
if [ "${STOP_WHEN_DONE:-0}" = 1 ] && [ "$VERIFIED" = yes ]; then
  say "=== ALL DONE and verified - stopping the box (STOP_WHEN_DONE=1 set at launch)"
  vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
else
  say "=== ALL STAGES FINISHED - box is IDLE and STILL BILLING; stop it by hand"
fi
