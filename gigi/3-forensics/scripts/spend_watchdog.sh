#!/usr/bin/env bash
# Stop every running Modal app once the OpenRouter key has spent CAP_USD since this started.
# Polls every 2 minutes; logs to results/logs/<stage>/watchdog.txt. Runs on the laptop (keep it awake).
#   FORENSICS_ENV_FILE=... CAP_USD=60 bash gigi/3-forensics/scripts/spend_watchdog.sh <stage>
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; LOG="$HERE/results/logs/$1"; mkdir -p "$LOG"
: "${FORENSICS_ENV_FILE:?}"; CAP_USD=${CAP_USD:?}
usage () { ( set -a; . "$FORENSICS_ENV_FILE"; set +a
  curl -s -m 20 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["data"]["usage"])' ) || echo ""; }
START=$(usage); echo "$(date '+%F %T') watchdog start: key spend \$$START, cap \$$CAP_USD" >> "$LOG/watchdog.txt"
cd "$HERE/agent-interp-envs"
while true; do
  sleep 120
  NOW=$(usage); [ -z "$NOW" ] && continue
  SPENT=$(python3 -c "print(round(float('$NOW') - float('$START'), 4))")
  echo "$(date '+%F %T') spent \$$SPENT" >> "$LOG/watchdog.txt"
  if python3 -c "import sys; sys.exit(0 if $SPENT >= $CAP_USD else 1)"; then
    echo "$(date '+%F %T') CAP REACHED: stopping every running Modal app" >> "$LOG/watchdog.txt"
    env -u VIRTUAL_ENV uv run modal app list --json | python3 -c "
import sys, json
for a in json.load(sys.stdin):
    if 'stopped' not in str(a.get('State', a.get('state', ''))).lower(): print(a.get('App ID') or a.get('app_id'))" |
      while read id; do env -u VIRTUAL_ENV uv run modal app stop "$id" >> "$LOG/watchdog.txt" 2>&1; echo "stopped $id" >> "$LOG/watchdog.txt"; done
    exit 0
  fi
done
