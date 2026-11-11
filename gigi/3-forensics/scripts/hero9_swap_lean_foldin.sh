#!/usr/bin/env bash
# Fold in the hero9 swap with the lean fetch (11 Nov 2026). Replaces hero9_swap_foldin.sh's whole-fleet fetch, which would
# fill the laptop's disk (see scripts/hero9_swap_fetch_lean.py). Unattended and restartable; every step skips done work:
#   1. lean fetch, pass "state" then pass "messages", each restarted if its heartbeat is silent for 15 min (max 6 tries);
#   2. wait until none of the swap's Modal apps is running (5 min checks, at most 8 h), then both passes again
#      (they skipped the fleets whose app was still running);
#   3. redact, grade (Opus; GRADE_CAP for the whole pass, CEIL on the key since 18 Oct), score, chart, cost.
# Logs to results/logs/hero9_swap/foldin.txt. Does NOT commit.
#   CEIL=7544 GRADE_CAP=1500 nohup caffeinate -i bash /abs/.../scripts/hero9_swap_lean_foldin.sh > /dev/null 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
L="$HERE/results/logs/hero9_swap"; RAW="$HERE/results/raw/hero9_swap"
export MODAL_PROFILE=${MODAL_PROFILE:-spar} MODAL_SECRET_SUFFIX=${MODAL_SECRET_SUFFIX:--gigi} MODAL_VOLUME_PREFIX=${MODAL_VOLUME_PREFIX:-gigi-forensics-}
: "${CEIL:?set CEIL}"; : "${GRADE_CAP:?set GRADE_CAP}"
say () { echo "$(date '+%F %T') [lean] $*" >> "$L/foldin.txt"; }

fetch_pass () {   # $1 = state | messages; restarts a stalled pass; returns 0 when a pass ends clean
  local try pid
  for try in 1 2 3 4 5 6; do
    touch "$L/fetch_lean.beat"
    (cd "$HERE/agent-interp-envs" && env -u VIRTUAL_ENV uv run python ../scripts/hero9_swap_fetch_lean.py "$1") >> "$L/fetch_lean_stdout.txt" 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      if [ $(( $(date +%s) - $(stat -f %m "$L/fetch_lean.beat") )) -gt 900 ]; then
        say "fetch $1: silent for 15 min, restarting (try $try)"; pkill -P $pid; kill $pid 2>/dev/null; sleep 5
        pkill -f "hero9_swap_fetch_lean.py $1"; break
      fi
    done
    wait $pid 2>/dev/null; local rc=$?
    grep -q "DISK LOW" <(tail -3 "$L/fetch_lean.txt") && { say "fetch $1 stopped: disk below 1 GB. Free space, then rerun this script"; exit 1; }
    [ $rc = 0 ] && { say "fetch $1 pass clean"; return 0; }
    say "fetch $1 try $try ended rc=$rc; retrying"; sleep 30
  done
  say "fetch $1: still failures after 6 tries; going on with what is here"; return 1
}

live_apps () {   # how many of the swap's apps (ap- ids in the launch logs) are not stopped; ? if the listing failed
  (cd "$HERE/agent-interp-envs" && env -u VIRTUAL_ENV uv run python - "$L" <<'PY' 2>> "$L/wait_errors.txt"
import json, pathlib, re, subprocess, sys
ids = set()
for log in pathlib.Path(sys.argv[1]).glob("*__*__*.log"):
    ids |= set(re.findall(r"ap-[A-Za-z0-9]+", log.read_text(errors="replace")))
try:
    apps = json.loads(subprocess.run(["modal", "app", "list", "--json"], capture_output=True, text=True, timeout=180).stdout)
except Exception:
    print("?"); sys.exit()
print(sum(1 for a in apps if (a.get("app_id") or a.get("App ID")) in ids and "stopped" not in str(a.get("state", a.get("State", ""))).lower()))
PY
  )
}

say "start (ceiling $CEIL, grade cap $GRADE_CAP for the whole grading pass)"
fetch_pass state; fetch_pass messages
for i in $(seq 1 96); do
  n=$(live_apps); [ "$n" = 0 ] && { say "all swap apps stopped"; break; }
  [ $((i % 6)) = 1 ] && say "waiting: ${n} swap apps still running"
  sleep 300
done
fetch_pass state; fetch_pass messages
du -sh "$RAW" | sed "s/^/$(date '+%F %T') [lean] raw size: /" >> "$L/foldin.txt"

(cd "$HERE" && FORENSICS_ENV_FILE="$ROOT/.env" python3 scripts/redact_key.py "$RAW") >> "$L/redact.txt" 2>&1 && REDACTED=1 || { REDACTED=0; say "REDACTION FAILED: grading skipped"; }
if [ "$REDACTED" = 1 ]; then
  say "grading"
  (cd "$HERE" && FORENSICS_ENV_FILE="$ROOT/.env" HERO2_SWAP=1 SWAP_HERO_NAME=hero9_swap GRADE_CAP_USD="$GRADE_CAP" TOTAL_CEILING_USD="$CEIL" \
      GRADE_LOG_DIR="$L/grading" python3 scripts/grade_modal.py) >> "$L/grade.txt" 2>&1 || say "grading exited non-zero (see grade.txt)"
  say "grading: $(tail -1 "$L/grading/grade.txt" 2>/dev/null)"
fi
(cd "$HERE" && python3 scripts/hero9_swap_score.py) >> "$L/score.txt" 2>&1 && say "scored: $(tail -1 "$L/score.txt")" || say "SCORE FAILED"
(cd "$HERE/agent-interp-envs" && env -u VIRTUAL_ENV uv run --with matplotlib python ../scripts/hero9_swap_chart.py) >> "$L/chart.txt" 2>&1 && say "chart: $(tail -1 "$L/chart.txt")" || say "CHART FAILED"
(cd "$ROOT" && FORENSICS_ENV_FILE="$ROOT/.env" python3 gigi/3-forensics/scripts/hero_usage.py 2>&1 | grep -E "key usage|since 18 Oct" | sed "s/^/$(date '+%F %T') after: /") >> "$L/cost.txt"
say "DONE (not committed: the parent session commits)"
