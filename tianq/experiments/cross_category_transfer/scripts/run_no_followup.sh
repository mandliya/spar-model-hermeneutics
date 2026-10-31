#!/usr/bin/env bash
# Laptop, CPU, after run_no_followup_extract.sh: the transfer matrix with no follow-up.
#
#   PATH="$HOME/.venvs/dyl-probe-repro/bin:$PATH" setsid nohup ../experiments/cross_category_transfer/scripts/run_no_followup.sh \
#       > scratch/logs/no_followup_driver.out 2>&1 < /dev/null &
#
#   1. a probe per category and read position, tested on the others
#      (experiments/cross_category_transfer/compare_no_followup.py --plan lists the runs)
#   2. the three matrices, from the stored results
#
# Each step is its own ./run.sh call, so each gets its own raw log on the Hub.
# One status line per step goes to scratch/logs/no_followup_status.log. Safe to
# kill and rerun: train_probe loads stored probes and skips stored tests.
set -euo pipefail
# Every step runs from dyl-probe-repro, the pipeline's folder (run.sh, scratch/).
cd "$(dirname "${BASH_SOURCE[0]}")/../../../dyl-probe-repro"

STATUS=scratch/logs/no_followup_status.log
status() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$STATUS"; }
step() { status "started: $*"; if "$@"; then status "ok: $*"; else status "FAILED: $*"; exit 1; fi; }

status "driver started (pid $$)"
mapfile -t runs < <(PYTHONPATH="$PWD:$PWD/.." python -m experiments.cross_category_transfer.compare_no_followup --plan)
for args in "${runs[@]}"; do
  # shellcheck disable=SC2086  # one line of arguments, split on purpose
  step ./run.sh train_probe $args
done
step ./run.sh compare_no_followup
status "driver finished"
