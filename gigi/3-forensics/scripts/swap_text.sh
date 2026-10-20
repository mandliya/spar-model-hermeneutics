#!/usr/bin/env bash
# The 17 Oct swap again, gemma's side only, with gemma in text-mode tool calling (agent.text_tools=true):
# every replay point of scripts/replay.sh (incidents, honest runs, and gemma's own runs as the control),
# same settings, REPS per point, batches of 10, built from the CURRENT tree (--use-current; the default rebuilds
# the run's 17 Oct commit, which has no text mode). Qwen's replays are unchanged and are not rerun.
# Results: results/raw/swap_text/<tag>/from_step<K>/gemma/; logs and spend: results/logs/swap_text/.
#   FORENSICS_ENV_FILE=... bash gigi/3-forensics/scripts/swap_text.sh [reps]
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ENVS="$HERE/agent-interp-envs"; REPS=${1:-20}
LOGS="$HERE/results/logs/swap_text"; mkdir -p "$LOGS"
: "${FORENSICS_ENV_FILE:?set FORENSICS_ENV_FILE to a file holding only OPENROUTER_API_KEY}"
GEMMA="agent.model=google/gemma-3-27b-it,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null,agent.text_tools=true"
Q=results/raw/screen/precommit_hook/precommit_hook/qwen-qwen3-next-80b-a3b-instruct/2026-10-17_15-20-35-423567
B=results/raw/base_qwen/precommit_hook/precommit_hook/qwen-qwen3-next-80b-a3b-instruct/2026-10-17_15-24-48-348561
G=results/raw/base_gemma/precommit_hook/precommit_hook/google-gemma-3-27b-it/2026-10-17_15-27-18-736069
POINTS="precommit_q1:$Q/run-5:8,7,6 precommit_q1m:$Q/run-5:3 precommit_q2:$B/run-18:3,2 precommit_h1:$Q/run-1:3
precommit_h2:$Q/run-2:3 precommit_h4:$Q/run-4:3 precommit_g19:$G/run-19:5,3 precommit_g2:$G/run-2:6,3
precommit_g5:$G/run-5:8,3 precommit_g7h:$G/run-7:3"
usage () { ( set -a; . "$FORENSICS_ENV_FILE"; set +a
  curl -s -m 20 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["data"]["usage"])' ); }
BEFORE=$(usage); echo "$(date '+%F %T') swap_text start, reps $REPS | key spend before: \$$BEFORE" >> "$LOGS/cost.txt"
for p in $POINTS; do
  tag=${p%%:*}; rest=${p#*:}; run=$HERE/${rest%%:*}; steps=${rest##*:}
  for K in ${steps//,/ }; do
    left=$REPS
    while [ "$left" -gt 0 ]; do
      n=$(( left < 10 ? left : 10 )); left=$(( left - n ))
      ( cd "$ENVS" && env -u VIRTUAL_ENV caffeinate -i uv run python scripts/resume.py "$run/step-$K" --use-current --count "$n" \
          --overrides "$GEMMA" --no-judge --env-file "$FORENSICS_ENV_FILE" --timeout 1800 \
          --results-dir "$HERE/results/raw/swap_text/$tag/from_step$K/gemma" ) >> "$LOGS/${tag}_from_step$K.log" 2>&1
      echo "$(date '+%T') $tag from step-$K: batch of $n done" >> "$LOGS/cost.txt"
    done
  done
done
AFTER=$(usage)
python3 -c "print(f'$(date "+%F %T") done | key spend after: \${float($AFTER):.4f} | this run: \${float($AFTER) - float($BEFORE):.4f}')" >> "$LOGS/cost.txt"
