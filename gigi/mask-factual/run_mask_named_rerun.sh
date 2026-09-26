#!/usr/bin/env bash
# Re-run ONLY mask-factual's named/eval_named conditions with the real
# category vocabulary (FOLLOWUP_NAMED_MASK / FOLLOWUP_EVAL_NAMED_MASK).
#
# The existing results/mask_factual_benchmark_conditions.jsonl is NOT touched:
# this writes to results/mask_factual_named_rerun.jsonl instead, because that
# file's named/eval_named rows are the old (duplicate-of-plain) measurement
# and this project does not overwrite anything in results/. The two are merged
# at analysis time, with the old rows kept as a record of what was there.
#
# 86 rows x 2 conditions = 172 generations. ~10 minutes. No API calls.
#
# Usage on the box:
#   cd /workspace/lies-experiment
#   nohup bash run_mask_named_rerun.sh > results/mask_rerun.log 2>&1 & disown

set -o pipefail
cd "$(dirname "$0")"
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }

say "=== mask-factual named/eval_named re-run ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true

if [ ! -f results/mask_factual_provided_conditions.csv ]; then
  say "!!! ABORT: results/mask_factual_provided_conditions.csv missing"
  exit 1
fi

if python 00_mask_factual_selfreport_benchmark.py \
      --conditions named eval_named \
      --out mask_factual_named_rerun; then
  say "--- re-run OK ---"
else
  say "!!! re-run FAILED — partial rows kept, script is resumable"
  exit 1
fi

N=$(grep -c . results/mask_factual_named_rerun.jsonl 2>/dev/null || echo 0)
say "rows written: $N (expected 172)"
say "=== COMPLETE ==="
