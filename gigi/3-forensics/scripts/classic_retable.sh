#!/usr/bin/env bash
# classic_retable.sh (run as /workspace/retable.sh) (17 Nov 2026, ~23:45 UTC): the fleet's per-run table covered only the 8 gap tasks (HERO_ALL_ENVS was not
# set), so the held-out check saw 0 Qwen runs. Archive the 8-task outputs, rebuild the table over all 28 tasks, re-run
# the selection and the held-out check. Nothing else.
set -uo pipefail
H=/workspace/forensics/gigi/3-forensics; cd "$H/agent-interp-envs"
A=$H/results/archive/classic_table_8tasks_17nov; mkdir -p "$A"
for f in results/classic_full_runs__qwen3.6-27b-bf16.csv results/classic_full_runs__qwen3.6-27b-bf16_per_run.csv \
         results/classic_selected_per_run.csv results/logs/classic/selection.txt; do
  [ -f "$H/$f" ] && mv "$H/$f" "$A/" && echo "archived $f"
done
STAGES=$(cat "$H/results/logs/classic/stages.txt")
HERO=1 HERO_ALL_ENVS=1 HERO_MODEL=qwen3.6-27b-bf16 HERO_STAGES="$STAGES" HERO_OUT_PREFIX=classic_full_runs__qwen3.6-27b-bf16 \
  env -u VIRTUAL_ENV uv run python ../scripts/full_runs_table.py --hero > /workspace/logs/classic_table2.log 2>&1; echo "table rc=$?"
python3 -c "
import csv, collections
r = list(csv.DictReader(open('$H/results/classic_full_runs__qwen3.6-27b-bf16_per_run.csv')))
c = collections.Counter(x['environment'] for x in r); print(len(r), 'rows,', len(c), 'tasks'); print(dict(sorted(c.items())))"
env -u VIRTUAL_ENV uv run python ../scripts/classic_select.py --stages "$STAGES" > /workspace/logs/classic_select2.log 2>&1; echo "select rc=$?"
cat "$H/results/logs/classic/selection.txt"
