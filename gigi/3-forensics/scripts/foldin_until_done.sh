#!/usr/bin/env bash
# Supervise scripts/fix_losses_foldin.sh until it reports "done" (28 Oct 2026). The fold-in runs on the laptop and is
# idempotent (fetched fleets stamped, graded runs skipped, scoring recomputed), so if a pass stalls because the Mac slept,
# the next pass finishes the rest. Up to 5 passes, 2 min apart; one line per pass in results/logs/fix_losses/supervisor.txt.
#   CEIL=1420 nohup caffeinate -i -s bash gigi/3-forensics/scripts/foldin_until_done.sh > /dev/null 2>&1 & disown
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; L="$HERE/results/logs/fix_losses"
say () { echo "$(date '+%F %T') $*" >> "$L/supervisor.txt"; }
MARK="${MARK:-RERUN 3}"   # the progress.txt marker after which a "done" counts
finished () { sed -n "/$MARK/,\$p" "$L/progress.txt" | grep -qE "^[0-9: -]+ (done|KEY FOUND|not committing)"; }
say "supervisor start (pid $$)"
for pass in 1 2 3 4 5; do
  while pgrep -f "bash $HERE/scripts/fix_losses_foldin.sh|bash scripts/fix_losses_foldin.sh" > /dev/null; do sleep 60; done
  finished && { say "fold-in reported done after pass $pass"; exit 0; }
  say "pass $pass: fold-in not done, relaunching"
  echo "$(date '+%F %T') $MARK (supervisor pass $pass)" >> "$L/progress.txt"
  CEIL="${CEIL:-1420}" bash "$HERE/scripts/fix_losses_foldin.sh" > "$L/foldin_pass$pass.out" 2>&1
  sleep 120
done
finished && say "fold-in reported done" || say "GAVE UP after 5 passes: read $L/progress.txt"
