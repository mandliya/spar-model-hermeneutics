#!/usr/bin/env bash
# GPU box: the substitutes read a Qwen model's conversations (core/model.py DYL_READER), for the probe
# faithfulness gap and the substitutes' diversity charts. Run after run_qwen_extract.sh has stored the
# target's own rows. Every Gigi category, no follow-up + neutral, as the target; nothing is generated.
#
#   ./logged.sh 'tmux new -d -s hermeneutics "DYL_MODEL=qwen3-next-80b ../experiments/hermeneutics/scripts/run_hermeneutics.sh"'   # 1 x 80-96 GB
#
# Per substitute: extract, then its in-category scores (compare_faithfulness), then its diversity
# chart and heatmaps. Then the target's own scores and the gaps. The environment is
# requirements-blackwell.txt in the image's /venv/main (Qwen3.5 needs transformers 5). One status line
# per step goes to scratch/logs/hermeneutics_status.log. Safe to kill and rerun: every step resumes.
set -euo pipefail
# Every step runs from dyl-probe-repro, the pipeline's folder (run.sh, scratch/).
cd "$(dirname "${BASH_SOURCE[0]}")/../../../dyl-probe-repro"
source /venv/main/bin/activate
export DYL_MODEL="${DYL_MODEL:?set DYL_MODEL to a key of dyl_repro/core/model.py MODELS}" HF_HOME=/workspace/hf
READERS="${READERS:-gemma-3-12b-it gemma-3-27b-it qwen3.5-27b}"

STATUS=scratch/logs/hermeneutics_status.log
mkdir -p scratch/logs
status() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$STATUS"; }
step() { status "started: $*"; if "$@"; then status "ok: $*"; else status "FAILED: $*"; exit 1; fi; }

status "driver started (pid $$)"
for reader in $READERS; do
  export DYL_READER=$reader
  for dataset in ascii_sandbagging sycophancy_answer sycophancy_feedback mask_factual mask_known_facts \
                 mask_statistics mask_continuations mask_disinformation unanswerable capture_the_secret alibi; do
    step ./run.sh extract --dataset "$dataset" --followup none neutral
  done
  step ./run.sh compare_faithfulness
  step ./run.sh compare_diversity
  step ./run.sh compare_diversity_matrices
done
unset DYL_READER
# GAPS=0: stop here (the target's own scores come from another box; run the two steps below once they exist).
[ "${GAPS:-1}" = 0 ] && { status "driver finished (GAPS=0)"; exit 0; }
step ./run.sh compare_faithfulness
# shellcheck disable=SC2086
step ./run.sh compare_faithfulness --gaps $READERS
status "driver finished"
