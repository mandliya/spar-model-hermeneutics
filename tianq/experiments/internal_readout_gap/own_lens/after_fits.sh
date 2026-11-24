#!/usr/bin/env bash
# Everything on the box after the two lens fits have finished,
# detached and resumable:
#   (nohup setsid bash after_fits.sh > /dev/null 2>&1 < /dev/null &)
# On the GPUs: the token-position reads (step 3, J space only). On the CPU at the same time: the re-reads of the
# saved restart-point activations through each lens, and their scores (step 2). Then the uploads, each with its
# completeness check. One status file: /workspace/logs/own_lens_status.txt. Logs are named by the experiment whose
# outputs a step writes, so shared/upload.py files them with it.
set -uo pipefail
cd "$(dirname "$0")"
E=own_lens
. ../shared/steps.sh
OUT=/workspace/out/internal-readout-gap
BASE=gemma-3-27b-it; FINAL=gemma-distilled-step96; QWEN=qwen3.6-27b
C=$OUT/own_lens/$BASE; D=$OUT/own_lens/$FINAL
CONTROL=$C/${BASE}_jacobian_lens.pt; OWN=$D/${FINAL}_jacobian_lens.pt
[ -f "$CONTROL" ] && [ -f "$OWN" ] || { echo "$(date -u +%FT%TZ) after_fits: a final lens is missing" >> "$STATUS"; exit 1; }

gpu_work() {   # step 3: the reads at several token positions
  cd ../distillation_similarity
  g0() { E=distillation_positions queue 0 "read_$QWEN:$PY read_positions.py --model $QWEN" \
           "read_$BASE:$PY read_positions.py --model $BASE --own_lens $CONTROL"; }
  g1() { E=distillation_positions queue 1 "read_$FINAL:$PY read_positions.py --model $FINAL --own_lens $OWN" &&
         cd ../lie_detection_gap &&
         E=lie_detection_positions queue 1 "read_${BASE}__read_by_$BASE:$PY read_reply_positions.py --target $BASE --model $BASE" \
           "read_${BASE}__read_by_$QWEN:$PY read_reply_positions.py --target $BASE --model $QWEN"; }
  side_by_side g0 g1 || return 1
  cd ../distillation_similarity
  # restart points too long for one GPU: each model again with both
  E=distillation_positions queue 0,1 "read_wide_$QWEN:$PY read_positions.py --model $QWEN" \
    "read_wide_$BASE:$PY read_positions.py --model $BASE --own_lens $CONTROL" \
    "read_wide_$FINAL:$PY read_positions.py --model $FINAL --own_lens $OWN" || return 1
  E=distillation_positions step positions_similarity $PY positions_similarity.py || return 1
  cd ../lie_detection_gap
  LIE_EXPERIMENT=lie_detection_positions E=lie_detection_positions step bundle $PY bundle.py || return 1
}

cpu_work() {   # step 2: the saved activations through each lens, then the scores
  export CUDA_VISIBLE_DEVICES=
  local nc nf n equal
  nc=$(($(wc -l < "$C/${BASE}_convergence.csv") - 1)); nf=$(($(wc -l < "$D/${FINAL}_convergence.csv") - 1))
  n=$(( nc < nf ? nc : nf )); equal=$(( n / 100 * 100 ))
  echo "$(date -u +%FT%TZ) fits stopped at $nc (control) and $nf (final checkpoint) prompts; equal-count snapshots at $equal" >> "$STATUS"
  local ce de
  ce=$($PY snapshots.py --fit_dir "$C" --find "$equal") && de=$($PY snapshots.py --fit_dir "$D" --find "$equal") || return 1
  # every cell: <model's saved activations> through <lens>, all re-read the same way (CPU), the published lens included
  local m
  for m in $BASE $FINAL; do
    step "reread_${m}_published" $PY reread.py --model $m --lens published --name published-reread || return 1
    step "reread_${m}_control" $PY reread.py --model $m --lens "$CONTROL" --name control-lens || return 1
    step "reread_${m}_final" $PY reread.py --model $m --lens "$OWN" --name final-lens || return 1
  done
  step "reread_${BASE}_control_equal" $PY reread.py --model $BASE --lens "$ce" --name control-lens-equal-prompts || return 1
  step "reread_${FINAL}_final_equal" $PY reread.py --model $FINAL --lens "$de" --name final-lens-equal-prompts || return 1
  # the final control check: our control fit against the published fit, and the two fits against each other
  step curve_final $PY compare_lenses.py curve --ours "$C/${BASE}_convergence.csv" || return 1
  step curve_final_vs_control $PY compare_lenses.py curve --ours "$D/${FINAL}_convergence.csv" --other "$C/${BASE}_convergence.csv" || return 1
  step words_control_vs_published $PY compare_lenses.py words --a $BASE.control-lens --b $BASE.published-reread || return 1
  step words_reread_vs_first_read $PY compare_lenses.py words --a $BASE.published-reread --b $BASE || return 1
  step words_final_lens_vs_control_on_final $PY compare_lenses.py words --a $FINAL.final-lens --b $FINAL.control-lens || return 1
  cd ../distillation_similarity
  local name versions
  for name in "own_lens:$BASE.control-lens,$FINAL.final-lens" \
              "published_lens_reread:$BASE.published-reread,$FINAL.published-reread" \
              "control_lens_for_both:$BASE.control-lens,$FINAL.control-lens" \
              "final_lens_for_both:$BASE.final-lens,$FINAL.final-lens" \
              "own_lens_equal_prompts:$BASE.control-lens-equal-prompts,$FINAL.final-lens-equal-prompts"; do
    versions=${name#*:}; name=${name%%:*}
    SIMILARITY_VERSIONS=$versions SIMILARITY_SUFFIX=_$name E=distillation_similarity step "similarity_$name" $PY similarity.py || return 1
    SIMILARITY_VERSIONS=$versions SIMILARITY_SUFFIX=_$name E=distillation_similarity step "where_it_moved_$name" $PY where_it_moved.py || return 1
  done
}

gpu_work & G=$!
( cpu_work ); RC=$?
wait $G; RG=$?
echo "$(date -u +%FT%TZ) after_fits: cpu work exit=$RC, gpu work exit=$RG" >> "$STATUS"

# step 5: uploads.
cp "$STATUS" "$OUT/own_lens/status_at_upload.txt"
FAILED=0
for e in own_lens distillation_similarity distillation_positions lie_detection_positions; do
  [ -d "$OUT/$e" ] || continue
  E=$e step upload $PY ../shared/upload.py --experiment $e || FAILED=1
done
echo "$(date -u +%FT%TZ) after_fits done: cpu=$RC gpu=$RG uploads_failed=$FAILED" >> "$STATUS"
# Everything worked and is on HF with nothing missing: the box goes after a grace period (touch /workspace/keep_box
# to keep it). Anything failed: the box stays up for a fix, until the dead-man switch (destroy_box.sh at ...).
if [ $RC -eq 0 ] && [ $RG -eq 0 ] && [ $FAILED -eq 0 ]; then
  bash destroy_box.sh grace 45 "all steps and HF completeness checks passed"
fi
