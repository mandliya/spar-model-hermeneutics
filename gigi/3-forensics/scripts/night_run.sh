#!/usr/bin/env bash
# The whole full-run sequence as one detached job: launch the Qwen environments not yet launched,
# then every gemma environment, wait until no Modal app is still running (runs cap at 4 h each),
# then mirror the volumes to results/raw/modal/ and stop the spend watchdog. Logs to
# results/logs/full/night.txt. Keep the laptop awake and on power until it says "done".
#   FORENSICS_ENV_FILE=... nohup caffeinate -ims bash gigi/3-forensics/scripts/night_run.sh &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; LOG="$HERE/results/logs/full"; mkdir -p "$LOG"
say () { echo "$(date '+%F %T') $*" >> "$LOG/night.txt"; }
ENVS="optimize_rendering_engine:5 optimize_rendering_engine_graded:5 grade_your_own:5 leaked_tests:5
puppeteer:5 async_conversion:5 revert_or_refactor:5 optimize_lru_cache:10 lucky_seed:10
lucky_seed_readable_grader:10 leaked_scorer:10 reference_extraction:10 copying_of_answers:10
dashboard_costly_failure:10 chess:20 tictactoe:20 tower_of_hanoi:20 funding_email:20
lazy_investigation:20 eval_tampering:20 sandbagging:20 currency_conversion:20 parse_duration:20"
for model in qwen3-next-80b gemma-3-27b; do
  say "launching $model"
  bash "$HERE/scripts/modal_full.sh" full "$model" $ENVS || say "launcher for $model exited non-zero"
done
cd "$HERE/agent-interp-envs"
running () { env -u VIRTUAL_ENV uv run modal app list --json 2>/dev/null | python3 -c "
import sys, json
print(sum('stopped' not in str(a.get('state', a.get('State', ''))).lower() for a in json.load(sys.stdin)))"; }
for i in $(seq 1 48); do   # 8 h at most
  n=$(running); say "apps still running: ${n:-?}"
  [ "$n" = 0 ] && break
  sleep 600
done
say "syncing volumes"
env -u VIRTUAL_ENV uv run python ../scripts/modal_sync.py >> "$LOG/night_sync.log" 2>&1 || say "sync exited non-zero"
pkill -f "spend_watchdog.sh full" && say "watchdog stopped"
say "done"
