#!/usr/bin/env bash
# classic_fleet.sh: the classic round's Qwen generation stage, run detached ON THE BOX (distill/PLAN_classic.md section 1;
# 17 Nov 2026). Needs the Qwen server up (serve_qwen.sh) and the gates passed (results/logs/classic/gate.txt, smokes).
#   setsid nohup bash /workspace/forensics/gigi/3-forensics/scripts/classic_fleet.sh > /workspace/logs/classic_fleet.out 2>&1 < /dev/null &
# Stages: launch 40 runs per task (46 in the 13 long training tasks) -> wait for every app -> lean fetch -> count complete
# runs -> top-up rounds (at most 2) -> per-run table (no Opus) -> upload to the private HF dataset. Never stops the box.
# Every step logs to /workspace/logs/stages.txt. Idempotent: modal_full.sh skips launches whose log exists.
set -uo pipefail
H=/workspace/forensics/gigi/3-forensics; cd "$H/agent-interp-envs"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
set -a; . /workspace/.env; set +a
S=/workspace/logs/stages.txt; say () { echo "$(date -u +%FT%TZ) $*" >> "$S"; }
TAG=qwen3.6-27b-bf16
LONG="async_conversion dashboard_costly_failure grade_your_own leaked_scorer leaked_tests optimize_lru_cache optimize_rendering_engine optimize_rendering_engine_graded precommit_hook puppeteer reference_extraction revert_or_refactor"
SHORT="chess coding coding_norules copying_of_answers currency_conversion eval_tampering funding_email lazy_investigation lucky_seed lucky_seed_readable_grader norvane parse_duration secret_number tictactoe tower_of_hanoi"

wait_stage () {   # until every app log of the stage has ended (completed, stopped or dead)
  local st=$1
  while true; do
    local n=0 done=0
    for f in "$H/results/logs/$st"/*__*.log; do
      [ -f "$f" ] || continue; n=$((n+1))
      if grep -qE "App completed|Stopping app|launch failed" "$f" || { tail -n 40 "$f" | grep -qE "ConnectionError|Could not connect to the Modal server" && ! pgrep -f "[r]esults/raw/$st/" >/dev/null; }; then done=$((done+1)); fi
    done
    say "$st: $done of $n apps ended"
    [ "$n" -gt 0 ] && [ "$done" -eq "$n" ] && return 0
    sleep 600
  done
}

fetch () {   # lean fetch (rollout.log, final/, last step) of the given stages
  for i in 1 2 3; do env -u VIRTUAL_ENV uv run python ../scripts/distill_fetch_lean.py "$1" >> /workspace/logs/classic_fetch.log 2>&1 && return 0; sleep 60; done
  say "FETCH FAILED for $1 (see classic_fetch.log)"
}

say "START classic fleet"
spec=""; for e in $SHORT; do spec="$spec $e:40"; done; for e in $LONG; do spec="$spec $e:46"; done
MAXC=5 GAP=12 MODAL_SECRETS=vllm-secret bash "$H/scripts/modal_full.sh" classic_qwen36 $TAG $spec
MAXC=3 GAP=12 MODAL_SECRETS=vllm-secret,openrouter-agent bash "$H/scripts/modal_full.sh" classic_qwen36 $TAG ai_rd:46
say "classic_qwen36 launched: $spec ai_rd:46"
wait_stage classic_qwen36; fetch classic_qwen36
STAGES=classic_qwen36
for r in 1 2; do
  need=$(env -u VIRTUAL_ENV uv run python ../scripts/classic_counts.py "$STAGES" --shortfall 2>>/workspace/logs/classic_fetch.log)
  env -u VIRTUAL_ENV uv run python ../scripts/classic_counts.py "$STAGES" > "$H/results/logs/classic/counts_after_round_$((r-1)).txt" 2>&1
  [ -z "$need" ] && { say "every task has 40 complete runs"; break; }
  say "top-up $r: $need"
  ai=$(echo "$need" | tr ' ' '\n' | grep '^ai_rd:' || true); rest=$(echo "$need" | tr ' ' '\n' | grep -v '^ai_rd:' | tr '\n' ' ')
  [ -n "$rest" ] && MAXC=5 GAP=12 MODAL_SECRETS=vllm-secret bash "$H/scripts/modal_full.sh" "classic_qwen36_t$r" $TAG $rest
  [ -n "$ai" ] && MAXC=3 GAP=12 MODAL_SECRETS=vllm-secret,openrouter-agent bash "$H/scripts/modal_full.sh" "classic_qwen36_t$r" $TAG $ai
  wait_stage "classic_qwen36_t$r"; fetch "classic_qwen36_t$r"; STAGES="$STAGES,classic_qwen36_t$r"
done
env -u VIRTUAL_ENV uv run python ../scripts/classic_counts.py "$STAGES" > "$H/results/logs/classic/counts_final.txt" 2>&1
say "counts: $(grep -c . "$H/results/logs/classic/counts_final.txt") lines in results/logs/classic/counts_final.txt"
HERO=1 HERO_MODEL=$TAG HERO_STAGES="$STAGES" HERO_OUT_PREFIX=classic_full_runs__$TAG \
  env -u VIRTUAL_ENV uv run python ../scripts/full_runs_table.py --hero >> /workspace/logs/classic_table.log 2>&1 && say "per-run table written" || say "PER-RUN TABLE FAILED (classic_table.log)"
for s in ${STAGES//,/ }; do
  hf upload model-hermeneutics/gigi-forensics-transcripts "$H/results/raw/$s" "distill/classic_v1/raw/$s" --repo-type dataset >> /workspace/logs/classic_upload.log 2>&1 \
    && say "uploaded $s to HF" || say "UPLOAD FAILED for $s (classic_upload.log)"
done
say "CLASSIC FLEET DONE: stages $STAGES"
