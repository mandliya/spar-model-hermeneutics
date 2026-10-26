#!/usr/bin/env bash
# ai_rd's swap for both models, paced with no laptop attached (26 Oct 2026). Same as aird_slow_rerun.sh, but the
# pacing is done by Modal itself: all 180 replays are created now, detached, and replay g (0..179) sleeps g * STEP
# seconds before it starts (resume_modal.py: STAGGER_START / STAGGER_STEP). One new replay every 90 s, each ~12 min:
# about 8 running at once, done in ~4.5 h. The laptop is needed only for the ~15 min of launching and, at the end,
# to fetch the results (scripts/aird_v2_finish.sh).
#   bash gigi/3-forensics/scripts/aird_detached_rerun.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ENVS="$HERE/agent-interp-envs"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
LOG="$HERE/results/logs/swap_aird_v2"; mkdir -p "$LOG"; STEP=${STEP:-90}
GEMMA="agent.model=google/gemma-3-27b-it,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null,agent.text_tools=true"
usage () { ( set -a; . "$HERE/../../.env"; set +a
  curl -s -m 20 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["data"]["usage"])' ); }
echo "$(date '+%F %T') detached rerun start (one replay every ${STEP}s) | key spend before: \$$(usage)" >> "$LOG/cost.txt"
python3 - "$HERE" > "$LOG/commands.txt" <<'PY'
import json, pathlib, sys
here = pathlib.Path(sys.argv[1])
for p in json.load(open(here / "results/swap_aird_plan.json")):
    run = here / p["run"]; point = f"{run.parent.name}_{run.name}_K{p['K']}"
    for who in ("qwen", "gemma"):
        print(point, who, run / f"step-{p['K']}", p["reps"])
PY
L=0
while read -r point who step n; do
  log="$LOG/ai_rd__${point}__${who}.log"; out="$HERE/results/raw/swap_aird_v2/$point/$who"
  ov=""; [ "$who" = gemma ] && ov="$GEMMA"
  ( cd "$ENVS" && STAGGER_START=$((L * n * STEP)) STAGGER_STEP=$STEP MODAL_SECRETS=openrouter-secret nohup env -u VIRTUAL_ENV \
      uv run modal run --detach scripts/resume_modal.py --step-dir "$step" --count "$n" --use-current --max-concurrent 0 \
      --no-judge --timeout 7200 --results-dir "$out" ${ov:+--overrides "$ov"} > "$log" 2>&1 & )
  sleep 45
  echo "$(date '+%F %T') launched $point $who x$n, starts at +$((L * n * STEP / 60)) min" >> "$LOG/cost.txt"
  L=$((L + 1))
done < "$LOG/commands.txt"
echo "$(date '+%F %T') all launched" >> "$LOG/cost.txt"
