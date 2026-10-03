#!/usr/bin/env bash
# Laptop, CPU, after run_new_gigi_categories.sh: John's two questions, end to end.
#
#   PATH="$HOME/.venvs/dyl-probe-repro/bin:$PATH" setsid nohup ./run_generalization.sh \
#       > scratch/logs/generalization_driver.out 2>&1 < /dev/null &
#
#   1. the released probe on the five new categories under Gigi's neutral and named
#      (the old six were scored before), for the outcome breakdown
#   2. a probe trained on every category under each wording, tested on the others
#      (cli/compare_generalization.py --plan lists the runs)
#   3. the matrices and the breakdown, from the stored results
#
# Each step is its own ./run.sh call, so each gets its own raw log on the Hub.
# One status line per step goes to scratch/logs/generalization_status.log. Safe
# to kill and rerun: scored categories are skipped, train_probe loads stored
# probes and skips stored tests.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

STATUS=scratch/logs/generalization_status.log
status() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$STATUS"; }
step() { status "started: $*"; if "$@"; then status "ok: $*"; else status "FAILED: $*"; exit 1; fi; }

declare -A NAMED=(
  [alibi]=named_alibi
  [mask_known_facts]=named_mask_belief
  [mask_statistics]=named_mask_belief
  [mask_continuations]=named_continuations
  [mask_disinformation]=named_disinformation
)
RELEASED=l_58_lm_0_01_ar_lr_hs59

status "driver started (pid $$)"
for dataset in alibi mask_known_facts mask_statistics mask_continuations mask_disinformation; do
  [[ -f "scratch/$dataset/neutral/results/$RELEASED.json" && -f "scratch/$dataset/${NAMED[$dataset]}/results/$RELEASED.json" ]] \
    || step ./run.sh evaluate --dataset "$dataset" --followup neutral "${NAMED[$dataset]}"
done

mapfile -t runs < <(python -m dyl_repro.cli.compare_generalization --plan)
for args in "${runs[@]}"; do
  # shellcheck disable=SC2086  # one line of arguments, split on purpose
  step ./run.sh train_probe $args
done

step ./run.sh compare_generalization
step ./run.sh compare_outcome_flags
status "driver finished"
