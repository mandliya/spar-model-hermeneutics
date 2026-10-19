#!/usr/bin/env bash
# The whole full-run sequence on Modal as one detached job: launch every Qwen environment, then every gemma
# environment (minus GEMMA_SKIP), wait until none of our Modal apps is still running (runs cap at 4 h each),
# then mirror our volumes to results/raw/modal/ and stop the spend watchdog. Logs to
# results/logs/<STAGE>/night.txt. Keep the laptop awake and on power until it says "done": a detached
# launch keeps only its last run alive if its local client dies.
#   FORENSICS_ENV_FILE=... MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics- \
#     STAGE=full_modal nohup caffeinate -ims bash gigi/3-forensics/scripts/night_run.sh &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; STAGE=${STAGE:-full}; LOG="$HERE/results/logs/$STAGE"; mkdir -p "$LOG"
say () { echo "$(date '+%F %T') $*" >> "$LOG/night.txt"; }
ENVS="optimize_rendering_engine:5 optimize_rendering_engine_graded:5 grade_your_own:5 leaked_tests:5
puppeteer:5 async_conversion:5 revert_or_refactor:5 optimize_lru_cache:10 lucky_seed:10
lucky_seed_readable_grader:10 leaked_scorer:10 reference_extraction:10 copying_of_answers:10
dashboard_costly_failure:10 chess:20 tictactoe:20 tower_of_hanoi:20 funding_email:20
lazy_investigation:20 eval_tampering:20 sandbagging:20 currency_conversion:20 parse_duration:20"
# gemma's tool calls are dropped by DeepInfra in these (README, Status): excluded for gemma until that is fixed
GEMMA_SKIP=${GEMMA_SKIP:-"currency_conversion funding_email"}
envs_for () {
  for spec in $ENVS; do
    [ "$1" = gemma-3-27b ] && [[ " $GEMMA_SKIP " == *" ${spec%%:*} "* ]] && continue
    echo "$spec"
  done
}
for attempt in $(seq 1 60); do   # a launch refused by Modal's workspace spend limit is retried every 10 min, 10 h at most
  for model in qwen3-next-80b gemma-3-27b; do
    say "launching $model"
    bash "$HERE/scripts/modal_full.sh" "$STAGE" "$model" $(envs_for "$model") || say "launcher for $model exited non-zero"
  done
  grep -lq "exceeded its spend limit" "$LOG"/*.log || break
  say "Modal workspace spend limit: $(grep -l "exceeded its spend limit" "$LOG"/*.log | wc -l | tr -d ' ') environments refused, retrying in 10 min"
  sleep 600
done
cd "$HERE/agent-interp-envs"
running () { env -u VIRTUAL_ENV uv run modal app list --json 2>/dev/null | python3 -c "
import sys, json   # ours only: the workspace may be shared
print(sum('stopped' not in str(a.get('state', a.get('State', ''))).lower() and '$HERE/configs/' in json.dumps(a)
          for a in json.load(sys.stdin)))"; }
for i in $(seq 1 48); do   # 8 h at most
  n=$(running); say "our apps still running: ${n:-?}"
  [ "$n" = 0 ] && break
  sleep 600
done
say "syncing volumes"
env -u VIRTUAL_ENV uv run python ../scripts/modal_sync.py >> "$LOG/night_sync.log" 2>&1 || say "sync exited non-zero"
pkill -f "spend_watchdog.sh $STAGE" && say "watchdog stopped"
say "done"
