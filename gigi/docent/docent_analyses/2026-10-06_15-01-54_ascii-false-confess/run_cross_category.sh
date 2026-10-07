#!/bin/bash
# Runs the cross-category analysis to completion, then builds its outputs and pushes them.
# A failed Docent step is retried; finished results are reused, so a retry only pays for
# the runs that had not finished. Logs: /Users/gg/Desktop/aisafety/docent_runs/
set -uo pipefail
REPO=/Users/gg/Desktop/aisafety/spar-model-hermeneutics
A=docent_analyses/2026-10-06_15-01-54_ascii-false-confess
LOG=/Users/gg/Desktop/aisafety/docent_runs
UV=/Users/gg/.local/bin/uv
cd "$REPO/gigi/docent"

ok=0
for attempt in 1 2 3 4 5 6; do
  echo "== analysis attempt $attempt, $(date)"
  if env -u VIRTUAL_ENV "$UV" run python -u "$A/cross_category_selfreport.py" > "$LOG/cross_category.attempt$attempt.log" 2>&1 \
     && grep -q "runs assigned" "$LOG/cross_category.attempt$attempt.log"; then ok=1; break; fi
  tail -3 "$LOG/cross_category.attempt$attempt.log"
  sleep 180
done
[ "$ok" = 1 ] || { echo "FAILED: analysis did not finish after 6 attempts, $(date)"; exit 1; }

set -e
echo "== building outputs, $(date)"
env -u VIRTUAL_ENV "$UV" run python -u "$A/build_cross_category.py"

echo "== committing, $(date)"
cd "$REPO"
git pull
git check-ignore -q .env
git add gigi/docent/
git commit -q -F - <<'MSG'
gigi: cross-category Docent analysis of gemma's self-report (all 11 categories)

Uploads every gemma category to Docent (01_upload.py gains a MASK belief builder
and --collection-id), reads all 5,554 reasoning runs for the mechanism behind
gemma's answer and whether it describes its own earlier answer correctly, and
adds an auto-generated cross-category section to DOCENT_RESULTS.md with charts,
tables and per-run CSV. Chart helpers move to charts_lib.py.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
git push
echo "ALL DONE, $(date)"
