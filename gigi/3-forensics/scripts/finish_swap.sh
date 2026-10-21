#!/usr/bin/env bash
# Finish the swap in every environment without supervision (21 Oct 2026), in order:
#   fetch what scoring needs -> top up replays lost to provider errors -> wait -> fetch -> wait for the running
#   grading round -> grade the new replays (Opus 5, then Opus 4.8 where it refuses) -> judge agreement check ->
#   rescore, chart and table -> upload the raw swap to HF -> commit and push the data (RESULTS.md prose by hand).
# Spend: grade_modal.py stops at TOTAL_CEILING_USD (1500 since the 18 Oct baseline); the swap_redo spend watchdog
# stops our Modal apps at the same ceiling. Log: results/logs/finish_swap.txt (ends with "done").
#   FORENSICS_ENV_FILE=... nohup caffeinate -ims bash gigi/3-forensics/scripts/finish_swap.sh &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; L="$HERE/results/logs/finish_swap.txt"; cd "$HERE"
: "${FORENSICS_ENV_FILE:?}"; export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
say () { echo "$(date '+%F %T') $*" >> "$L"; }
lean () { (cd agent-interp-envs && env -u VIRTUAL_ENV uv run python ../scripts/swap_fetch_lean.py) >> results/logs/swap_all/fetch_lean.log 2>&1; }
wait_for () { local t=0; while pgrep -f "$1" >/dev/null && [ $t -lt "$2" ]; do sleep 60; t=$((t + 60)); done; }
say "start"
wait_for swap_fetch_lean.py 3600; lean; say "fetch done"
python3 scripts/swap_all_score.py > /dev/null; say "top-ups: $(python3 scripts/swap_redo_plan.py)"
if [ -s results/logs/swap_all/commands_redo.txt ]; then
  bash scripts/swap_all.sh results/logs/swap_all/commands_redo.txt; say "top-ups launched"
  wait_for scripts/resume_modal.py 10800; say "top-ups finished (or 3 h passed)"
  lean; say "fetch after top-ups done"
fi
wait_for scripts/grade_modal.py 10800; say "earlier grading round finished"
SWAP_ALL=1 GRADE_CAP_USD=600 python3 scripts/grade_modal.py
SWAP_ALL=1 GRADE_CAP_USD=600 JUDGE_MODEL=claude-opus-4-8 python3 scripts/grade_modal.py; say "grading done"
python3 scripts/judge_agreement.py >> "$L" 2>&1; say "judge agreement done"
python3 scripts/swap_all_score.py > /dev/null; say "still missing after top-ups: $(python3 scripts/swap_redo_plan.py)"
(cd agent-interp-envs && env -u VIRTUAL_ENV uv run --with matplotlib python ../scripts/swap_all_chart.py \
  "Swapped into Qwen's runs, gemma copies Qwen's dishonesty but not its cheating") > /dev/null 2>&1; say "chart and table rebuilt"
(set -a; . "$HERE/../../.env"; set +a; cd agent-interp-envs && env -u VIRTUAL_ENV uv run --with huggingface_hub \
  python ../scripts/upload_raw.py swap_all) >> results/logs/upload_21oct.log 2>&1 && say "uploaded swap_all to HF" || say "HF upload failed"
cd "$HERE/../.." && git pull -q && git add gigi/3-forensics/results/swap_all.csv gigi/3-forensics/results/swap_all_table.md \
  gigi/3-forensics/results/swap_all_plan.json gigi/3-forensics/charts/swap_all.png gigi/3-forensics/results/judge_agreement.csv \
  gigi/3-forensics/results/logs/finish_swap.txt gigi/3-forensics/results/logs/swap_all/*.txt gigi/3-forensics/results/logs/grading/grade.txt \
  gigi/3-forensics/scripts && git commit -q -m "gigi: forensics swap: top-ups, remaining grading, judge agreement (data; RESULTS.md text follows)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && say "pushed" || say "push failed"
say "done"
