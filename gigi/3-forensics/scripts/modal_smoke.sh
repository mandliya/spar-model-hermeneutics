#!/usr/bin/env bash
# Smoke test of every environment on Modal: 1 run per model, capped at STEP_CAP steps (or the config's
# own lower cap), every run launched at once with --detach so the runs carry on when the laptop sleeps.
# Modal limits how fast apps can be created, so launches are spaced (GAP seconds) and a launch turned
# away by that limit is retried. swe_rebench_v2 is left out: its config is completed from the
# HF dataset at launch, which neither runner here does (task.cwd stays unresolved).
# Results stay on Modal's per-environment volumes; whatever lands while the laptop is awake is also
# downloaded to results/raw/modal_smoke/<env>/. One log per run in results/logs/modal_smoke/.
#   bash gigi/3-forensics/scripts/modal_smoke.sh [env ...]        (needs `modal token new` and the
#                                                                   Modal secret openrouter-secret)
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
STEP_CAP=${STEP_CAP:-30}; GAP=${GAP:-20}
ENVS=("$@")
[ ${#ENVS[@]} -eq 0 ] && ENVS=(tictactoe parse_duration currency_conversion sandbagging eval_tampering funding_email
  lazy_investigation tower_of_hanoi chess revert_or_refactor ai_rd async_conversion puppeteer
  optimize_lru_cache optimize_rendering_engine optimize_rendering_engine_graded dashboard_costly_failure
  grade_your_own lucky_seed lucky_seed_readable_grader leaked_scorer reference_extraction
  copying_of_answers leaked_tests)
LOG="$HERE/results/logs/modal_smoke"; mkdir -p "$LOG"
cd "$HERE/agent-interp-envs"
for e in "${ENVS[@]}"; do
  for m in qwen3-next-80b gemma-3-27b; do
    cfg="$HERE/configs/${e}__${m}.yaml"
    f="$LOG/${e}__${m}.log"
    # skip a run that is done or under way; relaunch one that never started (rate limit, or no log yet)
    [ -f "$f" ] && ! grep -q "rate limit exceeded" "$f" && continue
    cap=$(python3 -c "import re,sys; m=re.search(r'^  max_steps: *(\d+)', open(sys.argv[1]).read(), re.M); print(min(int(sys.argv[2]), int(m.group(1))) if m else sys.argv[2])" "$cfg" "$STEP_CAP")
    for try in 1 2 3 4 5 6; do
      MODAL_SECRETS=openrouter-secret nohup env -u VIRTUAL_ENV uv run modal run --detach scripts/run_modal.py --config "$cfg" \
        --count 1 --overrides "agent.max_steps=$cap" --timeout 3600 --no-judge \
        --results-dir "$HERE/results/raw/modal_smoke/$e" > "$f" 2>&1 &
      pid=$!; sleep "$GAP"
      grep -q "rate limit exceeded" "$f" || break
      echo "$(date '+%F %T') ${e}__${m}: rate-limited, retry $try" >> "$LOG/launch.txt"; sleep $((GAP * try))
    done
    echo "$(date '+%F %T') launched ${e}__${m} (max_steps $cap, pid $pid)" >> "$LOG/launch.txt"
  done
done
