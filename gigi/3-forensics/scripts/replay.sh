#!/usr/bin/env bash
# Replay one incident: resume a recorded run from checkpoint step-K, REPS times, with Qwen itself
# (the reference: how often does the target repeat its own choice?) and with gemma swapped in.
#   bash gigi/3-forensics/scripts/replay.sh <tag> <run-dir> <reps> <K>...
# step-K is the state AFTER step K, so resuming from step-(N-1) regenerates the decision at step N.
# Continuations go to results/raw/swap/<tag>/from_step<K>/<qwen|gemma>/; the key's spend before and
# after to results/logs/swap_<tag>/cost.txt. Batches of 10 (resume.py runs all of --count at once).
set -uo pipefail
TAG=$1; RUN=$(cd "$2" && pwd); REPS=$3; shift 3
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ENVS="$HERE/agent-interp-envs"
LOGS="$HERE/results/logs/swap_$TAG"; mkdir -p "$LOGS"
: "${FORENSICS_ENV_FILE:?set FORENSICS_ENV_FILE to a file holding only OPENROUTER_API_KEY}"
QWEN="agent.model=qwen/qwen3-next-80b-a3b-instruct,agent.temperature=0.7,agent.top_p=0.8,agent.top_k=20,agent.min_p=0"
GEMMA="agent.model=google/gemma-3-27b-it,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null"
usage () { ( set -a; . "$FORENSICS_ENV_FILE"; set +a
  curl -s -m 20 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["data"]["usage"])' ); }
BEFORE=$(usage); echo "$(date '+%F %T') swap=$TAG run=$RUN reps=$REPS from steps: $* | key spend before: \$$BEFORE" | tee "$LOGS/cost.txt"
for K in "$@"; do
  for who in qwen gemma; do
    ov=$QWEN; [ "$who" = gemma ] && ov=$GEMMA
    left=$REPS
    while [ "$left" -gt 0 ]; do
      n=$(( left < 10 ? left : 10 )); left=$(( left - n ))
      ( cd "$ENVS" && env -u VIRTUAL_ENV caffeinate -i uv run python scripts/resume.py "$RUN/step-$K" --count "$n" \
          --overrides "$ov" --no-judge --env-file "$FORENSICS_ENV_FILE" --timeout 1800 \
          --results-dir "$HERE/results/raw/swap/$TAG/from_step$K/$who" ) >> "$LOGS/from_step${K}_$who.log" 2>&1
      echo "$(date '+%T') from step-$K, $who: batch of $n done" | tee -a "$LOGS/cost.txt"
    done
  done
done
AFTER=$(usage)
python3 -c "print(f'$(date "+%F %T") done | key spend after: \${float($AFTER):.4f} | this swap: \${float($AFTER) - float($BEFORE):.4f}')" | tee -a "$LOGS/cost.txt"
