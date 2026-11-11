#!/usr/bin/env bash
# Laptop side of the hero9 swap handover (11 Nov 2026). Waits until a cloud session pushes handoff/CLOUD_READY
# (scripts/hero9_swap_cloud.sh), then stops the laptop's paced launcher (apps already launched keep running on
# Modal), builds the final handoff with scripts/hero9_swap_handoff.py, commits ONLY handoff/hero9_swap/ with the key-shape
# gate, and pushes. If the swap finishes launching on the laptop before the cloud is ready, it exits and does nothing.
#   nohup caffeinate -i bash scripts/hero9_swap_laptop_handover.sh > results/logs/hero9_swap/handover.out 2>&1 &
set -u
H="$(cd "$(dirname "$0")/.." && pwd)"; ROOT="$(cd "$H/../.." && pwd)"; L="$H/results/logs/hero9_swap"
say() { echo "$(date '+%F %T') [laptop] $*" | tee -a "$L/handover.txt"; }
cd "$ROOT" || exit 1
say "waiting for handoff/CLOUD_READY on origin/main"
until git fetch -q origin main 2>/dev/null && git cat-file -e origin/main:gigi/3-forensics/handoff/CLOUD_READY 2>/dev/null; do
  grep -q "all launched" "$L/launch.txt" 2>/dev/null && { say "the laptop launched everything first; no handover needed"; exit 0; }
  sleep 120
done
say "cloud is ready; stopping the laptop launcher"
pkill -f "hero9_swap_paced.sh" ; sleep 30
pgrep -f "hero9_swap_paced.sh" >/dev/null && { say "launcher still alive after kill; aborting handover"; exit 1; }
python3 "$H/scripts/hero9_swap_handoff.py" 2>&1 | tee -a "$L/handover.txt" | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || { say "handoff build failed; nothing pushed (restart the laptop launcher by hand if needed)"; exit 1; }
date '+%F %T laptop launcher stopped, handoff built' > "$H/handoff/hero9_swap/HANDOFF_DONE"
git pull -q --rebase --autostash || { say "git pull failed"; exit 1; }
git check-ignore -q .env || { say ".env not ignored: abort"; exit 1; }
git add "$H/handoff/hero9_swap" "$L/launch.txt" "$L/paced.txt" "$L/handover.txt"
if git diff --cached | grep -qE "sk-or-v1-[0-9a-f]{20,}|hf_[A-Za-z0-9]{30,}"; then   # key SHAPES only: a real key always matches one
  git reset -q; say "KEY SHAPE IN DIFF: nothing committed"; exit 1
fi
git commit -q -m "gigi: hero9 swap handed over to the cloud session (laptop launcher stopped)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && say "handoff pushed: $(git log --oneline -1)"
