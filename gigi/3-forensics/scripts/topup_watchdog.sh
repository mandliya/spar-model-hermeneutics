#!/usr/bin/env bash
# Stop the topup_gaps apps (the app IDs in its launch logs) once the key's spend since the 18 Oct baseline reaches CEIL.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; LOG="$HERE/results/logs/topup_gaps"; CEIL=${CEIL:-1480}
while true; do
  U=$( ( set -a; . "$HERE/../../.env"; set +a; curl -s -m 20 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" ) | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"]["usage"])' 2>/dev/null)
  if [ -n "$U" ]; then S=$(python3 -c "print(round($U - 131.477198523, 2))"); echo "$(date '+%F %T') since 18 Oct: \$$S" >> "$LOG/watchdog.txt"
    if python3 -c "import sys; sys.exit(0 if $S >= $CEIL else 1)"; then
      echo "$(date '+%F %T') CEILING \$$CEIL REACHED: stopping topup_gaps apps" >> "$LOG/watchdog.txt"
      for a in $(grep -aoh "ap-[A-Za-z0-9]*" "$LOG"/*.log | sort -u); do (cd "$HERE/agent-interp-envs" && MODAL_PROFILE=spar env -u VIRTUAL_ENV uv run modal app stop -y "$a" >/dev/null 2>&1); done
      exit 0; fi; fi
  sleep 180
done
