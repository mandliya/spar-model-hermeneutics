#!/usr/bin/env bash
# ai_rd's swap again with a scorer that still exists (26 Oct 2026). Its score.py asked OpenRouter for
# openai/gpt-3.5-turbo, a name no provider serves any more (404 from 25 Oct); now openai/gpt-3.5-turbo-16k (OpenAI's
# own endpoint, the pin in api_utils.py). score.py differs from the shipped file in that one line only. Both models
# are resumed at all 9 ai_rd points (6 incidents, 3 clean runs), 10 replays each, so both continue under the same
# scorer. The image's workspace now differs from the checkpoints' (score.py), so the resume runs with
# task.fs_baseline_mismatch=warn (the framework's opt-in for a deliberate branch into a changed image); before every
# checkpoint the agent had edited only notes.md and prompt.txt, so the restored workspace keeps the new score.py.
# The 21 Oct replays and their logs move to archive/ai_rd_gpt35_21oct/ (never deleted).
#   bash gigi/3-forensics/scripts/aird16k_rerun.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ENVS="$HERE/agent-interp-envs"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
LOG="$HERE/results/logs/swap_all"; ARCH_RAW="$HERE/results/raw/archive/ai_rd_gpt35_21oct"; ARCH_LOG="$HERE/results/logs/archive/ai_rd_gpt35_21oct"
mkdir -p "$ARCH_RAW" "$ARCH_LOG"
GEMMA="agent.model=google/gemma-3-27b-it,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null,agent.text_tools=true"
MISMATCH="task.fs_baseline_mismatch=warn"
usage () { ( set -a; . "$HERE/../../.env"; set +a
  curl -s -m 20 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["data"]["usage"])' ); }
echo "$(date '+%F %T') aird16k start | key spend before: \$$(usage)" >> "$LOG/aird16k_cost.txt"
python3 - "$HERE" > "$LOG/aird16k_commands.txt" <<'PY'
import json, pathlib, sys
here = pathlib.Path(sys.argv[1])
for p in json.load(open(here / "results/swap_aird_plan.json")):
    run = here / p["run"]; point = f"{run.parent.name}_{run.name}_K{p['K']}"
    for who in ("qwen", "gemma"):
        print(point, who, run / f"step-{p['K']}", p["reps"])
PY
while read -r point who step n; do
  out="$HERE/results/raw/swap_all/ai_rd/$point/$who"; log="$LOG/ai_rd__${point}__${who}.log"
  if [ -d "$out" ]; then mkdir -p "$ARCH_RAW/$point"; mv "$out" "$ARCH_RAW/$point/$who"; fi
  [ -f "$log" ] && mv "$log" "$ARCH_LOG/"
  ov="$MISMATCH"; [ "$who" = gemma ] && ov="$GEMMA,$MISMATCH"
  for try in 1 2 3 4 5 6; do
    ( cd "$ENVS" && MODAL_SECRETS=openrouter-secret nohup env -u VIRTUAL_ENV uv run modal run --detach scripts/resume_modal.py \
        --step-dir "$step" --count "$n" --use-current --max-concurrent 0 --no-judge --timeout 7200 \
        --results-dir "$out" --overrides "$ov" > "$log" 2>&1 & ); sleep 30
    grep -q "rate limit exceeded" "$log" || break
    sleep $((15 * try))
  done
  echo "$(date '+%F %T') launched ai_rd $point $who x$n" >> "$LOG/aird16k_cost.txt"
done < "$LOG/aird16k_commands.txt"
echo "$(date '+%F %T') all launched" >> "$LOG/aird16k_cost.txt"
