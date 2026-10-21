#!/usr/bin/env bash
# Fold the last long Qwen top-ups of the swap (21 Oct 2026) in, unattended: wait for their launchers (by PID,
# never by name pattern) -> fetch -> grade (Opus 5, then Opus 4.8 where it refuses) -> rescore -> chart and
# table (also written into RESULTS.md) -> HF upload of the raw swap -> commit and push -> stop the spend watchdog.
# The prose numbers that depend on the totals are updated by hand afterwards. Log: results/logs/finish_swap.txt.
#   FORENSICS_ENV_FILE=... nohup caffeinate -ims bash gigi/3-forensics/scripts/fold_topups.sh "<launcher PIDs>" "<watchdog PIDs>" &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; L="$HERE/results/logs/finish_swap.txt"; cd "$HERE"
: "${FORENSICS_ENV_FILE:?}"; export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
LAUNCHERS=$1; WATCHDOG=${2:-}
say () { echo "$(date '+%F %T') $*" >> "$L"; }
say "fold_topups start, waiting for launchers $LAUNCHERS"
t=0
while :; do
  alive=0; for p in $LAUNCHERS; do kill -0 "$p" 2>/dev/null && alive=1; done
  [ "$alive" = 0 ] && break
  [ "$t" -ge 9000 ] && { say "launchers still alive after 2.5 h, going on"; break; }
  sleep 60; t=$((t + 60))
done
say "top-up launchers done"
(cd agent-interp-envs && env -u VIRTUAL_ENV uv run python ../scripts/swap_fetch_lean.py) >> results/logs/swap_all/fetch_lean.log 2>&1
say "fetch done"
SWAP_ALL=1 GRADE_CAP_USD=100 python3 scripts/grade_modal.py
SWAP_ALL=1 GRADE_CAP_USD=100 JUDGE_MODEL=claude-opus-4-8 python3 scripts/grade_modal.py; say "grading done"
python3 scripts/swap_all_score.py > /dev/null; say "rescored"
(cd agent-interp-envs && env -u VIRTUAL_ENV uv run --with matplotlib python ../scripts/swap_all_chart.py \
  "In Qwen's place, gemma matches Qwen on the honesty probes but cheats far less on the coding tasks") > /dev/null 2>&1
python3 - <<'EOF'
import pathlib, time
p = pathlib.Path("RESULTS.md"); t = p.read_text()
start = t.index("| Category | Environment | Restart point | Qwen replayed | gemma swapped in |"); end = t.index("\n\n", start)
t = t[:start] + pathlib.Path("results/swap_all_table.md").read_text().rstrip("\n") + t[end:]
old = ("  - Still running at 19:40: 4 Qwen top-ups at final-decision points (lucky_seed_readable_grader ×3,\n"
       "    async_conversion ×1). They are added when they land.")
t = t.replace(old, f"  - The 4 long Qwen top-ups at final-decision points (lucky_seed_readable_grader ×3, async_conversion\n"
                   f"    ×1) are folded in ({time.strftime('%d %b %H:%M')}); the counts in this section's prose predate them.")
p.write_text(t)
EOF
say "chart and table rebuilt, RESULTS.md table replaced"
(set -a; . "$HERE/../../.env"; set +a; cd agent-interp-envs && env -u VIRTUAL_ENV uv run --with huggingface_hub \
  python ../scripts/upload_raw.py swap_all) >> results/logs/upload_21oct.log 2>&1 && say "uploaded swap_all to HF" || say "HF upload failed"
cd "$HERE/../.." && git pull -q && git add gigi/3-forensics/results/swap_all.csv gigi/3-forensics/results/swap_all_table.md \
  gigi/3-forensics/charts/swap_all.png gigi/3-forensics/RESULTS.md gigi/3-forensics/results/logs/finish_swap.txt \
  gigi/3-forensics/results/logs/grading/grade.txt gigi/3-forensics/results/logs/swap_all/*.txt gigi/3-forensics/scripts \
  && git commit -q -m "gigi: forensics swap: last long Qwen top-ups folded in (data, chart, table)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && say "pushed" || say "push failed"
[ -n "$WATCHDOG" ] && kill $WATCHDOG 2>/dev/null && say "spend watchdog stopped"
say "done"
