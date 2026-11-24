#!/usr/bin/env bash
# 17 Nov: the part of after_fits.sh that failed, redone. Qwen3.6's reads gave NaN under transformers 5.23.0 (the box's
# unpinned readout venv); /workspace/venvs/readout_qwen has the 13 Nov versions (transformers 5.22.2, peft 0.21.2)
# and reads it correctly. Gemma's reads are unaffected and stay in the readout venv.
#   (nohup setsid bash positions_fix.sh > /dev/null 2>&1 < /dev/null &)
# On GPU 0 now: Qwen3.6 and base gemma at the restart-point positions, then Qwen3.6 on the lie-detection rows.
# after_fits.sh is still using GPU 1 (final checkpoint, then gemma on the lie-detection rows); once it has ended,
# the wide pass, the position scores, the bundle and the uploads run here, and the box goes after a grace period.
set -uo pipefail
cd "$(dirname "$0")"
E=own_lens
. ../shared/steps.sh
PYQ=/workspace/venvs/readout_qwen/bin/python
OUT=/workspace/out/internal-readout-gap
BASE=gemma-3-27b-it; FINAL=gemma-distilled-step96; QWEN=qwen3.6-27b
CONTROL=$OUT/own_lens/$BASE/${BASE}_jacobian_lens.pt; OWN=$OUT/own_lens/$FINAL/${FINAL}_jacobian_lens.pt
RC=0
cd ../distillation_similarity
E=distillation_positions queue 0 "read_fixed_$QWEN:$PYQ read_positions.py --model $QWEN" \
  "read_$BASE:$PY read_positions.py --model $BASE --own_lens $CONTROL" || RC=1
cd ../lie_detection_gap
E=lie_detection_positions queue 0 "read_fixed_${BASE}__read_by_$QWEN:$PYQ read_reply_positions.py --target $BASE --model $QWEN" || RC=1
until grep -q "after_fits done" "$STATUS"; do sleep 30; done
cd ../distillation_similarity
E=distillation_positions queue 0,1 "read_wide_$QWEN:$PYQ read_positions.py --model $QWEN" \
  "read_wide_$BASE:$PY read_positions.py --model $BASE --own_lens $CONTROL" \
  "read_wide_$FINAL:$PY read_positions.py --model $FINAL --own_lens $OWN" || RC=1
E=distillation_positions step positions_similarity $PY positions_similarity.py || RC=1
cd ../lie_detection_gap
LIE_EXPERIMENT=lie_detection_positions E=lie_detection_positions step bundle $PY bundle.py || RC=1
cd ../own_lens
cp "$STATUS" "$OUT/own_lens/status_at_upload.txt"
FAILED=0
for e in own_lens distillation_similarity distillation_positions lie_detection_positions; do
  E=$e step upload_after_fix $PY ../shared/upload.py --experiment $e || FAILED=1
done
echo "$(date -u +%FT%TZ) positions_fix done: steps_failed=$RC uploads_failed=$FAILED" >> "$STATUS"
if [ $RC -eq 0 ] && [ $FAILED -eq 0 ]; then
  bash destroy_box.sh grace 45 "positions_fix: all steps and HF completeness checks passed"
fi
