#!/usr/bin/env bash
# Finish ai_rd and swe_rebench_v2 (launched 21 Oct 21:31 with the main key, the user's choice), unattended: wait for
# the four launches (log markers, never process names) -> copy both volumes off Modal -> redact keys from every output
# (scripts/redact_key.py) -> score (full_runs_table.py) and redraw the full-run charts -> HF upload (upload_raw.py
# refuses anything key-shaped anyway) -> commit and push the data; RESULTS.md prose by hand. Log: results/logs/full_key/run.txt
#   FORENSICS_ENV_FILE=... nohup caffeinate -ims bash gigi/3-forensics/scripts/finish_key.sh &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; LOG="$HERE/results/logs/full_key"; cd "$HERE"
: "${FORENSICS_ENV_FILE:?}"; export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
say () { echo "$(date '+%F %T') $*" >> "$LOG/run.txt"; }
done_all () {
  grep -q "App completed\|Stopping app" "$LOG/ai_rd__qwen3-next-80b.log" && grep -q "App completed\|Stopping app" "$LOG/ai_rd__gemma-3-27b.log" \
    && grep -q "^done:" "$LOG/swe_rebench_v2__qwen3-next-80b.log" && grep -q "^done:" "$LOG/swe_rebench_v2__gemma-3-27b.log"
}
say "finish_key start"
t=0; until done_all || [ "$t" -ge 14400 ]; do sleep 60; t=$((t + 60)); done
done_all && say "all four launches finished" || say "4 h passed; going on with what is there"
(cd agent-interp-envs && env -u VIRTUAL_ENV uv run python ../scripts/modal_sync.py gigi-forensics-ai-rd gigi-forensics-swe-rebench-v2) >> "$LOG/sync.log" 2>&1
say "volumes synced"
python3 scripts/redact_key.py results/raw/modal/gigi-forensics-ai-rd results/raw/modal/gigi-forensics-swe-rebench-v2 \
  results/raw/full_key "$LOG" >> "$LOG/redact.log" 2>&1; say "redacted: $(tail -1 "$LOG/redact.log")"
python3 scripts/full_runs_table.py > /dev/null && say "scored"
(cd agent-interp-envs && for s in full_runs_chart full_runs_overview full_runs_findings; do
   env -u VIRTUAL_ENV uv run --with matplotlib python ../scripts/$s.py > /dev/null 2>&1; done); say "charts redrawn"
(set -a; . "$HERE/../../.env"; set +a; cd agent-interp-envs && env -u VIRTUAL_ENV uv run --with huggingface_hub \
  python ../scripts/upload_raw.py modal) >> "$LOG/upload.log" 2>&1; say "HF: $(grep -a '^uploaded\|^REFUSED' "$LOG/upload.log" | tail -1)"
cd "$HERE/../.." && git pull -q && git add gigi/3-forensics/results/full_runs.csv gigi/3-forensics/charts \
  gigi/3-forensics/results/logs/full_key/*.txt gigi/3-forensics/scripts && git commit -q -m "gigi: forensics: ai_rd and swe_rebench_v2 full runs scored (outputs redacted); charts (RESULTS.md text follows)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && say "pushed" || say "push failed"
say "done"
