#!/usr/bin/env bash
# GPU box: extract the five categories added at Gigi's 0df3751 (alibi and the
# four MASK belief subtypes) with the three follow-ups of the transfer
# experiment: AISI's dyl, answered on-policy, and Gigi's neutral and named,
# his recorded answers prefilled.
#
#   ./logged.sh 'tmux new -d -s extract ../experiments/probe_vs_just_asking/scripts/run_new_gigi_categories.sh'
#
# Each dataset is its own ./run.sh call, so each gets its own raw log on the
# Hub. Safe to kill and rerun: extract skips rows already stored.
set -euo pipefail
# Every step runs from dyl-probe-repro, the pipeline's folder (run.sh, scratch/).
cd "$(dirname "${BASH_SOURCE[0]}")/../../../dyl-probe-repro"

declare -A NAMED=(
  [alibi]=named_alibi
  [mask_known_facts]=named_mask_belief
  [mask_statistics]=named_mask_belief
  [mask_continuations]=named_continuations
  [mask_disinformation]=named_disinformation
)

for dataset in alibi mask_known_facts mask_statistics mask_continuations mask_disinformation; do
  ./run.sh extract --dataset "$dataset" --followup dyl neutral "${NAMED[$dataset]}"
done
