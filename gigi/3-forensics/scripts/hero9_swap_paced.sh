#!/usr/bin/env bash
# Paced launcher for the hero9 symmetric swap (10 Nov 2026). hero9_swap.sh launched 68 of the 658 resumes at once and
# DeepInfra answered the ~680 concurrent continuations with 429s up to retry attempt 9 (the provider gives up at 14),
# so the rest is launched here in small batches, gated on how many swap containers are live on Modal. Idempotent: a
# resume whose log already shows a launch is skipped (same rule as hero9_swap.sh launch_one). When everything is
# launched it starts the fold-in exactly as hero9_swap.sh would have. Detached and sleep-safe:
#   nohup caffeinate -i bash scripts/hero9_swap_paced.sh <commands file> <cap USD> > results/logs/hero9_swap/paced.out 2>&1 &
# Tune while it runs by writing a number to results/logs/hero9_swap/MAX_TASKS (live containers allowed before the
# next batch; default 250) or BATCH (resumes per batch; default 8).
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
SWAPNAME=hero9_swap; LOG="$HERE/results/logs/$SWAPNAME"
export MODAL_PROFILE=${MODAL_PROFILE:-spar} MODAL_SECRET_SUFFIX=${MODAL_SECRET_SUFFIX:--gigi} MODAL_VOLUME_PREFIX=${MODAL_VOLUME_PREFIX:-gigi-forensics-}
CMDS=$1; CAP=$2
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG/paced.txt"; }

live_tasks() {  # live containers of our swap apps; -1 when the listing fails (then we wait rather than launch blind)
  (cd "$HERE/agent-interp-envs" && env -u VIRTUAL_ENV uv run modal app list --json 2>/dev/null) | python3 -c '
import json, sys
try: a = json.load(sys.stdin)
except Exception: print(-1); sys.exit()
print(sum(int(x.get("tasks") or 0) for x in a if "/results/raw/hero9_swap/" in (x.get("description") or "") and "ephemeral" in str(x.get("state"))))'
}

launch_one () {
  log=$1; shift
  [ -f "$log" ] && grep -q "Launching\|Resuming\|Checkpoint:" "$log" && ! grep -q "rate limit exceeded" "$log" && return 1
  for try in 1 2 3 4 5 6 7 8; do
    ( cd "$HERE/agent-interp-envs" && MODAL_SECRETS=${MODAL_SECRETS:-openrouter-secret} nohup env -u VIRTUAL_ENV uv run modal run --detach \
        scripts/resume_modal.py "$@" > "$log" 2>&1 & ); sleep 25
    grep -q "rate limit exceeded" "$log" || break
    sleep $((10 * try))
  done
  echo "$(date '+%F %T') launched $(basename "$log")" >> "$LOG/launch.txt"
  return 0
}

total=$(wc -l < "$CMDS" | tr -d ' ')
say "paced launcher start: $total resumes in $(basename "$CMDS"), cap \$$CAP"
n_launched=0; n_skipped=0; in_batch=0
while IFS= read -r line; do
  [ -n "$line" ] || continue
  eval "set -- $line"
  log=$1
  if [ -f "$log" ] && grep -q "Launching\|Resuming\|Checkpoint:" "$log" && ! grep -q "rate limit exceeded" "$log"; then n_skipped=$((n_skipped+1)); continue; fi
  BATCH=$(cat "$LOG/BATCH" 2>/dev/null || echo 8); MAX=$(cat "$LOG/MAX_TASKS" 2>/dev/null || echo 250)
  if [ $in_batch -ge "$BATCH" ] || [ $n_launched -eq 0 ]; then
    in_batch=0
    while :; do
      t=$(live_tasks)
      if [ "$t" -ge 0 ] && [ "$t" -le "$MAX" ]; then say "live containers $t <= $MAX: next batch of $BATCH ($n_launched launched so far, $n_skipped skipped)"; break; fi
      say "live containers $t > $MAX: waiting 3 min"; sleep 180
    done
  fi
  launch_one "$@" && { n_launched=$((n_launched+1)); in_batch=$((in_batch+1)); }
done < "$CMDS"
say "all launched: $n_launched here, $n_skipped already launched before"
echo "$(date '+%F %T') all launched ($(basename "$CMDS"), paced)" >> "$LOG/launch.txt"

# the fold-in, exactly as hero9_swap.sh starts it
spent_now=$(FORENSICS_ENV_FILE="$ROOT/.env" python3 "$HERE/scripts/hero_usage.py" 2>/dev/null | grep "since 18 Oct" | grep -oE '\$[0-9][0-9,]*\.[0-9]+' | head -1 | tr -d '$,')
CEIL=${CEIL:-$(python3 -c "print(round(float('${spent_now:-0}') + float('$CAP') + 100))")} GRADE_CAP=${GRADE_CAP:-60} bash "$HERE/scripts/hero9_swap_foldin.sh"
echo "$(date '+%F %T') fold-in started (ceiling ${CEIL:-see foldin.txt})" >> "$LOG/launch.txt"
say "fold-in started (ceiling $CEIL)"
