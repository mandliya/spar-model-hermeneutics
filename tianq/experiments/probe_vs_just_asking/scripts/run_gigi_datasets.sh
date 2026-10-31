#!/usr/bin/env bash
# GPU box: extract, then score, every one of Gigi's datasets with every
# follow-up it asks in the master list (dyl_repro/datasets/followups.json).
#
#   ./logged.sh 'tmux new -d -s gigi ../experiments/probe_vs_just_asking/scripts/run_gigi_datasets.sh'
#
# ASCII runs only the two follow-ups it gained from Gigi's list; its other
# fourteen were extracted and scored before. Gigi's recorded answers are
# prefilled; the other follow-ups are answered on-policy at temperature 0. Each
# step is its own ./run.sh call, so each gets its own raw log on the Hub. Safe
# to kill and rerun: extract skips rows already stored, and evaluate rescores
# from what is stored.
set -euo pipefail
# Every step runs from dyl-probe-repro, the pipeline's folder (run.sh, scratch/).
cd "$(dirname "${BASH_SOURCE[0]}")/../../../dyl-probe-repro"

DATASETS=(sycophancy_answer sycophancy_feedback mask_factual unanswerable capture_the_secret)
ASCII_NEW=(eval_plain eval_named_sandbagging)

./run.sh extract --dataset ascii_sandbagging --followup "${ASCII_NEW[@]}"
for dataset in "${DATASETS[@]}"; do
  ./run.sh extract --dataset "$dataset" --followup all
done
./run.sh evaluate --dataset ascii_sandbagging --followup "${ASCII_NEW[@]}"
for dataset in "${DATASETS[@]}"; do
  ./run.sh evaluate --dataset "$dataset" --followup all
done
