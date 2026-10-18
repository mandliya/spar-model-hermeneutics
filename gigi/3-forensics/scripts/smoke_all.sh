#!/usr/bin/env bash
# Smoke test of every environment: 1 run per model, capped at STEP_CAP steps, a few environments at a
# time (a laptop runs them), stopping for good once the OpenRouter key has spent CAP_USD since the
# start. Each environment is its own stage, results/raw/smoke_all/<env>/, so a failing one stops
# nothing else.
#   FORENSICS_ENV_FILE=... bash gigi/3-forensics/scripts/smoke_all.sh [env ...]     (default: all)
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
: "${FORENSICS_ENV_FILE:?set FORENSICS_ENV_FILE to a file holding only OPENROUTER_API_KEY}"
CAP_USD=${CAP_USD:-5}; export STEP_CAP=${STEP_CAP:-30}; GROUP=${GROUP:-3}
ENVS=("$@")
[ ${#ENVS[@]} -eq 0 ] && ENVS=(tictactoe parse_duration currency_conversion sandbagging eval_tampering funding_email
  lazy_investigation tower_of_hanoi chess revert_or_refactor ai_rd async_conversion puppeteer
  optimize_lru_cache optimize_rendering_engine optimize_rendering_engine_graded dashboard_costly_failure
  grade_your_own lucky_seed lucky_seed_readable_grader leaked_scorer reference_extraction
  copying_of_answers leaked_tests)   # swe_rebench_v2: x86-only images, not runnable on Apple Silicon (its README)
LOG="$HERE/results/logs/smoke_all"; mkdir -p "$LOG"

usage () {
  ( set -a; . "$FORENSICS_ENV_FILE"; set +a
    curl -s -m 20 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
      | python3 -c 'import sys, json; print(json.load(sys.stdin)["data"]["usage"])' )
}
START=$(usage); echo "$(date '+%F %T') smoke_all start | key spend \$$START | cap \$$CAP_USD | step cap $STEP_CAP" | tee -a "$LOG/progress.txt"

for ((i = 0; i < ${#ENVS[@]}; i += GROUP)); do
  NOW=$(usage)
  if python3 -c "import sys; sys.exit(0 if float('$NOW') - float('$START') >= $CAP_USD else 1)"; then
    echo "$(date '+%F %T') STOP: spent \$$(python3 -c "print(round(float('$NOW') - float('$START'), 4))") >= cap; not run: ${ENVS[*]:$i}" | tee -a "$LOG/progress.txt"
    exit 0
  fi
  PIDS=()
  for e in "${ENVS[@]:$i:$GROUP}"; do
    bash "$HERE/scripts/run_stage.sh" "smoke_all/$e" 1 1 "$e"__qwen3-next-80b.yaml "$e"__gemma-3-27b.yaml > "$LOG/$e.stage.log" 2>&1 &
    PIDS+=($!)
  done
  for p in "${PIDS[@]}"; do wait "$p"; done
  echo "$(date '+%F %T') done: ${ENVS[*]:$i:$GROUP} | spent so far \$$(python3 -c "print(round(float('$(usage)') - float('$START'), 4))")" | tee -a "$LOG/progress.txt"
done
echo "$(date '+%F %T') smoke_all finished | spent \$$(python3 -c "print(round(float('$(usage)') - float('$START'), 4))")" | tee -a "$LOG/progress.txt"
