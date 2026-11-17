#!/usr/bin/env bash
# classic_fleet2.sh: relaunch of the classic round's Qwen generation, run detached ON THE BOX (distill/PLAN_classic.md,
# amendment of 17 Nov 2026). Replaces classic_fleet.sh, stopped at 08:45 UTC.
#   setsid nohup bash /workspace/forensics/gigi/3-forensics/scripts/classic_fleet2.sh > /workspace/logs/classic_fleet2.out 2>&1 < /dev/null &
# Why: the first fleet ran ~68 runs at once. Their histories (~1.5M tokens) did not fit in the server's 1.06M-token cache,
# so every step re-read its whole history (cache hits 8%, 2,904 preemptions), steps took minutes, and runs were cut by the
# 1 h / 4 h limits and by request timeouts. 16 tasks are re-run from scratch (results/logs/classic/set_aside.tsv); the
# other 12 keep their classic_qwen36 runs.
# Load now: 32 runs at once (~0.7M tokens), each task's share sized to its hero9 length, then 37 once the last two apps
# of the first fleet have ended. Server health (cache hits, preemptions, cache in use) goes to stages.txt every 10 min.
#   r2  (at start): the 16 tasks; the 5 short ones get half their runs, one at a time
#   r2b (once lucky_seed and lucky_seed_readable_grader of classic_qwen36 have ended, at most 90 min): the other halves
#   then: wait for every app -> lean fetch -> counts -> top-ups (at most 2) -> per-run table -> HF upload.
# Never stops the box.
set -uo pipefail
H=/workspace/forensics/gigi/3-forensics; cd "$H/agent-interp-envs"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
set -a; . /workspace/.env; set +a
S=/workspace/logs/stages.txt; say () { echo "$(date -u +%FT%TZ) $*" >> "$S"; }
TAG=qwen3.6-27b-bf16
SMALL="copying_of_answers:20 leaked_scorer:23 optimize_rendering_engine_graded:23 tower_of_hanoi:20 funding_email:20"

alive () { pgrep -f "results/raw/$1/$2\$" >/dev/null; }   # that stage's launcher for that task is still running
live () {   # how many of a stage's launchers are still running
  local n=0 f e
  for f in "$H/results/logs/$1"/*__*.log; do [ -f "$f" ] || continue; e=$(basename "$f"); alive "$1" "${e%%__*}" && n=$((n+1)); done
  echo $n
}
metrics () {   # "queries hits generated preemptions cache% running waiting epoch", or nothing if the server is down
  local m; m=$(curl -s -m 10 localhost:8000/metrics | awk '
    /^vllm:prefix_cache_queries_total/ {q+=$2} /^vllm:prefix_cache_hits_total/ {h+=$2}
    /^vllm:generation_tokens_total/ {g+=$2} /^vllm:num_preemptions_total/ {p+=$2}
    /^vllm:kv_cache_usage_perc/ {k=$2} /^vllm:num_requests_running/ {r=$2} /^vllm:num_requests_waiting[{ ]/ {w=$2; n=1}
    END {if (n) printf "%.0f %.0f %.0f %.0f %.0f %.0f %.0f", q, h, g, p, 100*k, r, w}')
  [ -n "$m" ] && echo "$m $(date +%s)"
}
PREV=""
health () {   # cache hits, output speed and preemptions since the last call
  local cur q0 h0 g0 p0 t0 q1 h1 g1 p1 k r w t1 x
  cur=$(metrics); [ -n "$cur" ] || { say "server: no metrics (is vllm up?)"; return 0; }
  read q1 h1 g1 p1 k r w t1 <<< "$cur"
  if [ -n "$PREV" ]; then
    read q0 h0 g0 p0 x x x t0 <<< "$PREV"
    local dt=$((t1 - t0)) dq=$((q1 - q0)); [ $dt -gt 0 ] || dt=1
    local hit=0; [ $dq -gt 0 ] && hit=$((100 * (h1 - h0) / dq))
    say "server, last $((dt / 60)) min: cache hits ${hit}%, $(((g1 - g0) / dt)) tokens/s generated, $((p1 - p0)) preemptions, cache ${k}% in use, $r requests running, $w waiting"
  fi
  PREV=$cur
}
fetch () {   # lean fetch (rollout.log, final/, last step) of the given stages
  for i in 1 2 3; do env -u VIRTUAL_ENV uv run python ../scripts/distill_fetch_lean.py "$1" >> /workspace/logs/classic_fetch.log 2>&1 && return 0; sleep 60; done
  say "FETCH FAILED for $1 (see classic_fetch.log)"
}
counts () { env -u VIRTUAL_ENV uv run python ../scripts/classic_counts.py "$@" 2>>/workspace/logs/classic_fetch.log; }

say "START classic fleet 2: 16 tasks re-run (set_aside.tsv), 32 runs at once"
health
MAXC=4 GAP=12 MODAL_SECRETS=vllm-secret bash "$H/scripts/modal_full.sh" classic_qwen36_r2 $TAG puppeteer:46
MAXC=3 GAP=12 MODAL_SECRETS=vllm-secret bash "$H/scripts/modal_full.sh" classic_qwen36_r2 $TAG reference_extraction:46 leaked_tests:46
MAXC=2 GAP=12 MODAL_SECRETS=vllm-secret bash "$H/scripts/modal_full.sh" classic_qwen36_r2 $TAG optimize_lru_cache:46 revert_or_refactor:46 \
  dashboard_costly_failure:46 grade_your_own:46 async_conversion:46 optimize_rendering_engine:46 precommit_hook:46
MAXC=3 GAP=12 MODAL_SECRETS=vllm-secret,openrouter-agent bash "$H/scripts/modal_full.sh" classic_qwen36_r2 $TAG ai_rd:46
MAXC=1 GAP=12 MODAL_SECRETS=vllm-secret bash "$H/scripts/modal_full.sh" classic_qwen36_r2 $TAG $SMALL
say "classic_qwen36_r2 launched: $(ls "$H/results/logs/classic_qwen36_r2"/*__*.log | wc -l) of 16 apps"

T0=$(date +%s); R2B=0
while true; do
  sleep 600; health
  if [ $R2B = 0 ] && { { ! alive classic_qwen36 lucky_seed && ! alive classic_qwen36 lucky_seed_readable_grader; } || [ $(($(date +%s) - T0)) -ge 5400 ]; }; then
    MAXC=1 GAP=12 MODAL_SECRETS=vllm-secret bash "$H/scripts/modal_full.sh" classic_qwen36_r2b $TAG $SMALL
    say "classic_qwen36_r2b launched: the other halves of $SMALL, one at a time"; R2B=1
  fi
  a=$(live classic_qwen36); b=$(live classic_qwen36_r2); c=$(live classic_qwen36_r2b)
  say "apps still running: classic_qwen36 $a, r2 $b of 16, r2b $c of 5"
  [ $R2B = 1 ] && [ "$a" -eq 0 ] && [ "$b" -eq 0 ] && [ "$c" -eq 0 ] && break
done

STAGES=classic_qwen36,classic_qwen36_r2,classic_qwen36_r2b
for s in ${STAGES//,/ }; do fetch $s; done
counts "$STAGES" > "$H/results/logs/classic/counts_after_r2.txt"
say "counts after the relaunch: $(tail -n 1 "$H/results/logs/classic/counts_after_r2.txt")"
for r in 1 2; do
  need=$(counts "$STAGES" --shortfall)
  [ -z "$need" ] && { say "every task has 40 complete runs"; break; }
  ntask=$(echo "$need" | wc -w); m=2; [ "$ntask" -gt 12 ] && m=1   # keep the top-up load at or under ~24 runs at once
  say "top-up $r ($m at a time per task): $need"
  ai=$(echo "$need" | tr ' ' '\n' | grep '^ai_rd:' || true); rest=$(echo "$need" | tr ' ' '\n' | grep -v '^ai_rd:' | tr '\n' ' ')
  [ -n "${rest// /}" ] && MAXC=$m GAP=12 MODAL_SECRETS=vllm-secret bash "$H/scripts/modal_full.sh" "classic_qwen36_t$r" $TAG $rest
  [ -n "$ai" ] && MAXC=$m GAP=12 MODAL_SECRETS=vllm-secret,openrouter-agent bash "$H/scripts/modal_full.sh" "classic_qwen36_t$r" $TAG $ai
  while [ "$(live "classic_qwen36_t$r")" -gt 0 ]; do sleep 600; health; done
  fetch "classic_qwen36_t$r"; STAGES="$STAGES,classic_qwen36_t$r"
done
counts "$STAGES" > "$H/results/logs/classic/counts_final.txt"
say "final counts: $(tail -n 1 "$H/results/logs/classic/counts_final.txt") (results/logs/classic/counts_final.txt)"
HERO=1 HERO_MODEL=$TAG HERO_STAGES="$STAGES" HERO_OUT_PREFIX=classic_full_runs__$TAG \
  env -u VIRTUAL_ENV uv run python ../scripts/full_runs_table.py --hero >> /workspace/logs/classic_table.log 2>&1 && say "per-run table written" || say "PER-RUN TABLE FAILED (classic_table.log)"
for s in ${STAGES//,/ }; do
  hf upload model-hermeneutics/gigi-forensics-transcripts "$H/results/raw/$s" "distill/classic_v1/raw/$s" --repo-type dataset >> /workspace/logs/classic_upload.log 2>&1 \
    && say "uploaded $s to HF" || say "UPLOAD FAILED for $s (classic_upload.log)"
done
hf upload model-hermeneutics/gigi-forensics-transcripts "$H/results/logs/classic" "distill/classic_v1/logs" --repo-type dataset >> /workspace/logs/classic_upload.log 2>&1 \
  && say "uploaded results/logs/classic to HF" || say "UPLOAD FAILED for results/logs/classic"
echo "$STAGES" > "$H/results/logs/classic/stages.txt"
say "CLASSIC FLEET DONE: stages $STAGES"
