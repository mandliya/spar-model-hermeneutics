#!/usr/bin/env bash
# The last steps of the swap, without waiting for the final long top-ups (they are folded in when they land):
# grade every downloaded replay (Opus 5, then Opus 4.8 where it refuses) -> judge agreement -> rescore, chart and
# table -> HF upload of the raw swap -> commit and push the data. Log: results/logs/finish_swap.txt (ends "done").
#   FORENSICS_ENV_FILE=... nohup caffeinate -ims bash gigi/3-forensics/scripts/finish_now.sh &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; L="$HERE/results/logs/finish_swap.txt"; cd "$HERE"
: "${FORENSICS_ENV_FILE:?}"; export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
say () { echo "$(date '+%F %T') $*" >> "$L"; }
say "finish_now start"
SWAP_ALL=1 GRADE_CAP_USD=600 python3 scripts/grade_modal.py
SWAP_ALL=1 GRADE_CAP_USD=600 JUDGE_MODEL=claude-opus-4-8 python3 scripts/grade_modal.py; say "grading done"
python3 scripts/judge_agreement.py >> "$L" 2>&1; say "judge agreement done"
python3 scripts/swap_all_score.py > /dev/null; say "still missing: $(python3 scripts/swap_redo_plan.py)"
(cd agent-interp-envs && env -u VIRTUAL_ENV uv run --with matplotlib python ../scripts/swap_all_chart.py \
  "Swapped into Qwen's runs, gemma copies Qwen's dishonesty but not its cheating") > /dev/null 2>&1; say "chart and table rebuilt"
(set -a; . "$HERE/../../.env"; set +a; cd agent-interp-envs && env -u VIRTUAL_ENV uv run --with huggingface_hub \
  python ../scripts/upload_raw.py swap_all) >> results/logs/upload_21oct.log 2>&1 && say "uploaded swap_all to HF" || say "HF upload failed"
cd "$HERE/../.." && git pull -q && git add gigi/3-forensics/results/swap_all.csv gigi/3-forensics/results/swap_all_table.md \
  gigi/3-forensics/results/swap_all_plan.json gigi/3-forensics/charts/swap_all.png gigi/3-forensics/results/judge_agreement.csv \
  gigi/3-forensics/results/logs/finish_swap.txt gigi/3-forensics/results/logs/swap_all/*.txt gigi/3-forensics/results/logs/grading/grade.txt \
  gigi/3-forensics/scripts && git commit -q -m "gigi: forensics swap: top-ups graded, judge agreement, rescored (data; RESULTS.md text follows)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && say "pushed" || say "push failed"
say "done"
