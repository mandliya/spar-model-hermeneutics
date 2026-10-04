#!/usr/bin/env bash
# GPU box: John's "probe without asking 'did you lie?'". Every Gigi category with
# the follow-up "none": the conversation ends at Gemma's recorded reply, which is
# run in full and kept at core/prefill.py REPLY_POSITIONS. Nothing is generated.
#
#   ./logged.sh 'tmux new -d -s extract ./run_no_followup_extract.sh'
#
# One ./run.sh call for all categories would load the model once, but each dataset
# is its own call so each gets its own raw log on the Hub, as before. Safe to kill
# and rerun: extract skips rows already stored.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

for dataset in ascii_sandbagging sycophancy_answer sycophancy_feedback mask_factual mask_known_facts \
               mask_statistics mask_continuations mask_disinformation unanswerable capture_the_secret alibi; do
  ./run.sh extract --dataset "$dataset" --followup none
done
