#!/usr/bin/env bash
# games/alibi end to end. No `set -e`: the final stop always runs.
# Stopping preserves /workspace, so it is always safe.
cd /workspace/lies-experiment
source /venv/main/bin/activate
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }

say "=== FIRST RUN (500 rows) ==="
python 00_alibi.py && say "--- first run OK ---" || say "!!! first run FAILED (resumable)"

say "=== BACKUP first-run results to HF ==="
python upload_files.py alibi results/alibi_conditions.csv results/alibi_conditions.jsonl

say "=== GATE ==="
if python 02_analyse_alibi.py --gate; then
  say "=== 12 FOLLOW-UP QUESTIONS ==="
  python 00_alibi_selfreport_benchmark.py && say "--- 12 questions OK ---" || say "!!! 12 questions FAILED (resumable)"
  say "=== ANALYSIS ==="
  python 02_analyse_alibi.py || say "!!! analysis failed (data safe)"
  say "=== BACKUP 12-question results to HF ==="
  python upload_files.py alibi results/alibi_benchmark_conditions.csv results/alibi_benchmark_conditions.jsonl
else
  say "!!! GATE FAILED - skipping the 12 questions; first-run results stand alone"
fi

say "=== BACKUP activations to HF ==="
# this box is fresh and only holds alibi tensors, so one tarball of all of
# them - a per-id-pattern glob would miss ids that are bare numbers ("63")
bash upload_activations.sh alibi '*.pt' || say "!!! activation upload failed (tensors safe on /workspace)"

say "=== COMPLETE - stopping instance ==="
bash stop_self.sh
