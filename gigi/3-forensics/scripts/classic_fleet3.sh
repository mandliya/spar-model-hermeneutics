#!/usr/bin/env bash
# classic_fleet3.sh: paced finish of the classic round's Qwen generation, run detached ON THE BOX (distill/PLAN_classic.md,
# amendment of 17 Nov 2026). Replaces classic_fleet2.sh, stopped at 09:25 UTC.
#   setsid nohup bash /workspace/forensics/gigi/3-forensics/scripts/classic_fleet3.sh > /workspace/logs/classic_fleet3.out 2>&1 < /dev/null &
# Why: 32-37 runs at once filled the server again by 09:20 (cache hits 21%, requests waiting): the long tasks' histories
# grow towards the 49k window, so a load that fits at the start does not fit 20 minutes later. 9 apps were stopped
# (set_aside.tsv); 12 r2 apps keep running, 20 runs at once.
# Now the load is paced: the stopped tasks go into a queue of small chunks that run one at a time (one app each, stage
# classic_qwen36_r3_<k> = chunk k), longest tasks first. A chunk starts only while (a) the runs in flight across every
# stage stay at or under CAP=20 (scripts/classic_sched.py) and (b) the server's last 10 minutes were healthy: cache
# hits >= 70% and nothing waiting. Health is logged every 5 minutes; launches are held while it is not healthy, and the
# cap moves with it (-2 when not healthy, down to 14; +1 when hits >= 80% and the cache is at most 60% full, up to 26).
# Then: lean fetch -> counts (set-aside pairs left out) -> top-ups through the same queue (at most 2 rounds) -> per-run
# table -> HF upload -> "CLASSIC FLEET DONE: stages ..." (classic_after.sh waits for that line). Never stops the box.
set -uo pipefail
H=/workspace/forensics/gigi/3-forensics; cd "$H/agent-interp-envs"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
set -a; . /workspace/.env; set +a
S=/workspace/logs/stages.txt; say () { echo "$(date -u +%FT%TZ) $*" >> "$S"; }
TAG=qwen3.6-27b-bf16; CAP=20; Q=/workspace/logs/classic_queue.txt
BASE="classic_qwen36,classic_qwen36_r2,classic_qwen36_r2b"   # stages before this script, in launch order

metrics () {   # "queries hits generated preemptions cache% running waiting epoch", or nothing if the server is down
  local m; m=$(curl -s -m 10 localhost:8000/metrics | awk '
    /^vllm:prefix_cache_queries_total/ {q+=$2} /^vllm:prefix_cache_hits_total/ {h+=$2}
    /^vllm:generation_tokens_total/ {g+=$2} /^vllm:num_preemptions_total/ {p+=$2}
    /^vllm:kv_cache_usage_perc/ {k=$2} /^vllm:num_requests_running/ {r=$2} /^vllm:num_requests_waiting[{ ]/ {w=$2; n=1}
    END {if (n) printf "%.0f %.0f %.0f %.0f %.0f %.0f %.0f", q, h, g, p, 100*k, r, w}')
  [ -n "$m" ] && echo "$m $(date +%s)"
}
PREV=""; HEALTHY=1
health () {   # logs the last window; HEALTHY=1 when cache hits >= 70% and nothing is waiting
  local cur q0 h0 g0 p0 t0 q1 h1 g1 p1 k r w t1 x
  cur=$(metrics); [ -n "$cur" ] || { say "server: no metrics (is vllm up?)"; HEALTHY=0; return 0; }
  read q1 h1 g1 p1 k r w t1 <<< "$cur"
  if [ -n "$PREV" ]; then
    read q0 h0 g0 p0 x x x t0 <<< "$PREV"
    local dt=$((t1 - t0)) dq=$((q1 - q0)); [ $dt -gt 0 ] || dt=1
    local hit=100; [ $dq -gt 0 ] && hit=$((100 * (h1 - h0) / dq))
    HEALTHY=1; { [ $hit -lt 70 ] || [ "$w" -gt 0 ]; } && HEALTHY=0
    if [ $HEALTHY = 0 ]; then CAP=$((CAP > 15 ? CAP - 2 : 14))   # unhealthy: hold launches and lower the cap
    elif [ $hit -ge 80 ] && [ "$k" -le 60 ]; then CAP=$((CAP < 26 ? CAP + 1 : 26)); fi   # room to spare: one more
    say "server, last $((dt / 60)) min: cache hits ${hit}%, $(((g1 - g0) / dt)) tokens/s generated, $((p1 - p0)) preemptions, cache ${k}% in use, $r requests running, $w waiting; cap now $CAP$([ $HEALTHY = 0 ] && echo ', NOT HEALTHY: launches held')"
  fi
  PREV=$cur
}
stages_all () { echo "$BASE$(cut -d' ' -f1 "$Q.done" 2>/dev/null | awk '!s[$0]++' | sed 's/^/,/' | tr -d '\n')"; }
live () { python3 ../scripts/classic_sched.py live "$(stages_all)"; }
apps () { python3 ../scripts/classic_sched.py apps "$(stages_all)"; }
fetch () {
  for i in 1 2 3; do env -u VIRTUAL_ENV uv run python ../scripts/distill_fetch_lean.py "$1" >> /workspace/logs/classic_fetch.log 2>&1 && return 0; sleep 60; done
  say "FETCH FAILED for $1 (see classic_fetch.log)"
}
counts () { env -u VIRTUAL_ENV uv run python ../scripts/classic_counts.py "$@" 2>>/workspace/logs/classic_fetch.log; }
chunks () {   # chunks <stage prefix> <task> <runs>: queue lines "<prefix>_<k> <task> <n>", at most 6 runs per chunk
  local p=$1 e=$2 n=$3 k=1 c
  while [ "$n" -gt 0 ]; do c=$(( (n + (n + 5) / 6 - 1) / ((n + 5) / 6) )); echo "${p}_$k $e $c"; n=$((n - c)); k=$((k + 1)); done
}
run_queue () {   # launch the queue as health and CAP allow; return once it is empty and nothing is in flight
  local next st e n l
  while true; do
    health
    l=$(live)
    while [ -s "$Q" ] && [ $HEALTHY = 1 ] && [ $((l + 1)) -le $CAP ]; do
      next=$(head -n 1 "$Q"); read st e n <<< "$next"
      sec=vllm-secret; [ "$e" = ai_rd ] && sec=vllm-secret,openrouter-agent   # ai_rd's runs also call its capped agent key
      MAXC=1 GAP=12 MODAL_SECRETS=$sec bash "$H/scripts/modal_full.sh" "$st" $TAG "$e:$n"
      echo "$next" >> "$Q.done"; sed -i 1d "$Q"; l=$((l + 1))
      say "launched $st $e x$n (one at a time); $l runs in flight, $(grep -c . "$Q" || true) chunks queued"
    done
    [ -s "$Q" ] || [ "$(apps)" -gt 0 ] || return 0
    say "in flight: $l runs in $(apps) apps (cap $CAP); $(grep -c . "$Q" || true) chunks queued"
    sleep 300
  done
}

say "START classic fleet 3: paced, cap $CAP runs at once"
: > "$Q.done"
{ for e in puppeteer leaked_tests reference_extraction optimize_lru_cache; do chunks classic_qwen36_r3 $e 46; done
  chunks classic_qwen36_r3 leaked_scorer 23; chunks classic_qwen36_r3 copying_of_answers 20
  chunks classic_qwen36_r3 optimize_rendering_engine_graded 23
  chunks classic_qwen36_r3 tower_of_hanoi 20; chunks classic_qwen36_r3 funding_email 20; } > "$Q"   # longest chunks first
say "queue: $(wc -l < "$Q") chunks, $(awk '{s+=$3} END {print s}' "$Q") runs (puppeteer, leaked_tests, reference_extraction, optimize_lru_cache x46; leaked_scorer, optimize_rendering_engine_graded x23; copying_of_answers, tower_of_hanoi, funding_email x20)"
run_queue

STAGES=$(stages_all)
for s in ${STAGES//,/ }; do fetch "$s"; done
counts "$STAGES" > "$H/results/logs/classic/counts_after_r3.txt"
say "counts after the paced launch: $(tail -n 1 "$H/results/logs/classic/counts_after_r3.txt") (cut: column 4)"
for r in 1 2; do
  need=$(counts "$STAGES" --shortfall)
  [ -z "$need" ] && { say "every task has 40 complete runs"; break; }
  say "top-up $r: $need"
  : > "$Q"
  for x in $need; do chunks "classic_qwen36_t$r" "${x%%:*}" "${x##*:}" >> "$Q"; done
  before=$(cut -d' ' -f1 "$Q.done" | sort -u)
  run_queue
  STAGES=$(stages_all)
  for s in $(cut -d' ' -f1 "$Q.done" | sort -u); do echo "$before" | grep -qx "$s" || fetch "$s"; done
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
