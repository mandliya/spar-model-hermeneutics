#!/usr/bin/env bash
# Two more checks for the 22 Oct call's double-check (26 Oct 2026), detached (no laptop needed once launched):
#  1. Rendering engine: gemma's 60 replays at Qwen's 6 incident points rerun with the parser fix (10 of the 60 had
#     stumbled on unreadable calls just before ending). The superseded fleets and logs move to archive/parser_cut_render_26oct/.
#     Graded later with the other new replays (grade_modal.py, SWAP_ALL=1).
#  2. ai_rd alone: both models' 10 full runs again, with the parser fix and the scorer alias (stage full_aird_v2), paced
#     with staggered starts (one start every 5 min, Qwen and gemma alternating) so the scorer is never crowded.
#   bash gigi/3-forensics/scripts/soundness_extra.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ENVS="$HERE/agent-interp-envs"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
LOG="$HERE/results/logs/soundness_extra"; mkdir -p "$LOG"
ARCH_RAW="$HERE/results/raw/archive/parser_cut_render_26oct"; ARCH_LOG="$HERE/results/logs/archive/parser_cut_render_26oct"; mkdir -p "$ARCH_RAW" "$ARCH_LOG"
GEMMA="agent.model=google/gemma-3-27b-it,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null,agent.text_tools=true"
usage () { ( set -a; . "$HERE/../../.env"; set +a
  curl -s -m 20 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["data"]["usage"])' ); }
echo "$(date '+%F %T') soundness_extra start | key spend before: \$$(usage)" >> "$LOG/cost.txt"
python3 - "$HERE" > "$LOG/render_points.txt" <<'PY'
import json, pathlib, sys
here = pathlib.Path(sys.argv[1])
for p in json.load(open(here / "results/swap_all_plan.json")):
    if p["env"] == "optimize_rendering_engine" and p["incident"]:
        run = here / p["run"]; print(f"{run.parent.name}_{run.name}_K{p['K']}", run / f"step-{p['K']}")
PY
while read -r point step; do
  out="$HERE/results/raw/swap_all/optimize_rendering_engine/$point/gemma"; log="$HERE/results/logs/swap_all/optimize_rendering_engine__${point}__gemma.log"
  [ -d "$out" ] && { mkdir -p "$ARCH_RAW/$point"; mv "$out" "$ARCH_RAW/$point/gemma"; }
  [ -f "$log" ] && mv "$log" "$ARCH_LOG/"
  ( cd "$ENVS" && MODAL_SECRETS=openrouter-secret nohup env -u VIRTUAL_ENV uv run modal run --detach scripts/resume_modal.py \
      --step-dir "$step" --count 10 --use-current --max-concurrent 0 --no-judge --timeout 14400 \
      --results-dir "$out" --overrides "$GEMMA" > "$log" 2>&1 & ); sleep 40
  echo "$(date '+%F %T') rendering $point gemma x10 launched" >> "$LOG/cost.txt"
done < "$LOG/render_points.txt"
STAGGER_START=0 STAGGER_STEP=600 MAXC=0 GAP=30 bash "$HERE/scripts/modal_full.sh" full_aird_v2 qwen3-next-80b ai_rd:10
STAGGER_START=300 STAGGER_STEP=600 MAXC=0 GAP=30 bash "$HERE/scripts/modal_full.sh" full_aird_v2 gemma-3-27b ai_rd:10
echo "$(date '+%F %T') all launched" >> "$LOG/cost.txt"
