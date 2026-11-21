#!/usr/bin/env bash
# 17 Nov, exploratory (asked for after the headline result): the lens readouts as probability distributions, for
# Qwen3.6, base gemma and the final checkpoint, through the published lens and through each gemma's own lens.
#   (nohup setsid bash distributions.sh > /dev/null 2>&1 < /dev/null &)
# CPU only. Holds the box (/workspace/keep_box) until its own upload has passed, then releases it.
set -uo pipefail
cd "$(dirname "$0")"
E=distillation_similarity
. ../shared/steps.sh
touch /workspace/keep_box
export CUDA_VISIBLE_DEVICES=
PYQ=/workspace/venvs/readout_qwen/bin/python
OUT=/workspace/out/internal-readout-gap
BASE=gemma-3-27b-it; FINAL=gemma-distilled-step96; QWEN=qwen3.6-27b
CONTROL=$OUT/own_lens/$BASE/${BASE}_jacobian_lens.pt; OWN=$OUT/own_lens/$FINAL/${FINAL}_jacobian_lens.pt
cd ../readout_diffing/jspace
RC=0
step distributions_qwen $PYQ lens_distributions.py --model $QWEN --lens published --name published || RC=1
step distributions_base_published $PY lens_distributions.py --model $BASE --lens published --name published || RC=1
step distributions_base_control $PY lens_distributions.py --model $BASE --lens "$CONTROL" --name control-lens || RC=1
step distributions_final_published $PY lens_distributions.py --model $FINAL --lens published --name published || RC=1
step distributions_final_own $PY lens_distributions.py --model $FINAL --lens "$OWN" --name final-lens || RC=1
echo "$(date -u +%FT%TZ) distributions done: steps_failed=$RC" >> "$STATUS"
