#!/usr/bin/env bash
# Run one stage of rollouts: every config given, COUNT runs each, in parallel, detached-safe.
#   bash gigi/3-forensics/scripts/run_stage.sh <stage> <count> <max_concurrent_per_config> <config.yaml>...
# Writes raw runs to results/raw/<stage>/, one log per config to results/logs/<stage>/, and the
# OpenRouter key's spend before and after to results/logs/<stage>/cost.txt (measured, not estimated).
# Keeps the Mac awake while it runs (caffeinate). Keys come from $FORENSICS_ENV_FILE, a file that
# holds only OPENROUTER_API_KEY (never the repo's whole .env).
# Optional STEP_CAP=<n>: cap each run at min(n, the config's own max_steps) steps (smoke tests).
set -uo pipefail
STAGE=$1; COUNT=$2; CONC=$3; shift 3
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ENVS="$HERE/agent-interp-envs"
LOGS="$HERE/results/logs/$STAGE"; mkdir -p "$LOGS" "$HERE/results/raw/$STAGE"
: "${FORENSICS_ENV_FILE:?set FORENSICS_ENV_FILE to a file holding only OPENROUTER_API_KEY}"

usage () {   # the key's lifetime spend in dollars
  ( set -a; . "$FORENSICS_ENV_FILE"; set +a
    curl -s -m 20 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
      | python3 -c 'import sys, json; print(json.load(sys.stdin)["data"]["usage"])' )
}
BEFORE=$(usage); echo "$(date '+%F %T') stage=$STAGE count=$COUNT configs=$# | key spend before: \$$BEFORE" | tee "$LOGS/cost.txt"

PIDS=()
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  OV=""
  if [ -n "${STEP_CAP:-}" ]; then
    OV="agent.max_steps=$(python3 -c "import re,sys; m=re.search(r'^  max_steps: *(\d+)', open(sys.argv[1]).read(), re.M); print(min(int(sys.argv[2]), int(m.group(1))) if m else sys.argv[2])" "$HERE/configs/$name.yaml" "$STEP_CAP")"
  fi
  ( cd "$ENVS" && env -u VIRTUAL_ENV caffeinate -i uv run python scripts/run.py "$HERE/configs/$name.yaml" \
      --count "$COUNT" --max-concurrent "$CONC" --no-judge --env-file "$FORENSICS_ENV_FILE" ${OV:+--overrides "$OV"} \
      --results-dir "$HERE/results/raw/$STAGE" --timeout 3600 ) > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
  echo "started $name (pid $!)${OV:+ with $OV}" | tee -a "$LOGS/cost.txt"
done
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=$((FAIL + 1)); done
AFTER=$(usage)
python3 -c "print(f'$(date "+%F %T") done | key spend after: \${float($AFTER):.4f} | this stage: \${float($AFTER) - float($BEFORE):.4f} | configs that exited non-zero: $FAIL')" | tee -a "$LOGS/cost.txt"
