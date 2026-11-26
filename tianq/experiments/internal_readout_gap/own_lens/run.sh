#!/usr/bin/env bash
# Fresh J-lenses for base gemma (the control) and a distilled checkpoint, on the box, detached and resumable
# Phases, in this order:
#   (nohup setsid bash run.sh gate > /dev/null 2>&1 < /dev/null &)    # 3 prompts on the final checkpoint; the lens must load
#   (nohup setsid bash run.sh fetch > /dev/null 2>&1 < /dev/null &)   # the earlier reads of the restart points, from HF
#   (nohup setsid bash run.sh fits > /dev/null 2>&1 < /dev/null &)    # the two full fits, one GPU each, with snapshots
#   (nohup setsid bash run.sh fit48 > /dev/null 2>&1 < /dev/null &)   # the 48-step checkpoint's lens on GPU $GPU48
# On a one-GPU box, for one checkpoint (FINAL names it; the default is the swap distillation's last):
#   (FINAL=gemma-classic-step92 nohup setsid bash run.sh gate fit_final upload_final > /dev/null 2>&1 < /dev/null &)
# Steps, logs and the status file: ../shared/steps.sh. Lenses land in /workspace/out/internal-readout-gap/own_lens/<model>.
set -uo pipefail
cd "$(dirname "$0")"
E=own_lens
. ../shared/steps.sh
LENSES=/workspace/out/internal-readout-gap/own_lens
FINAL=${FINAL:-gemma-distilled-step96}
BASE=google/gemma-3-27b-it      # the published lens's own model id; its slug names the folder gemma-3-27b-it

watch() {  # watch <folder name>: the snapshot watcher of that fit, and the pusher of its snapshots and resume copies to HF
  ( $PY snapshots.py --fit_dir "$LENSES/$1" >> "/workspace/logs/${E}_snapshots_$1.log" 2>&1 & )
  ( $PY push_fit_progress.py --fit "$1" >> "/workspace/logs/${E}_push_fit_progress_$1.log" 2>&1 & )
}
fit() {  # fit <model id or folder> <gpu> <folder name>: the fit, watched
  mkdir -p "$LENSES/$3"
  watch "$3"
  step "fit_$3" bash fit.sh "$1" "$2" "$LENSES/$3"
}
merged() {  # merged <distilled tag> <gpu>: the checkpoint as a model folder
  CUDA_VISIBLE_DEVICES=$2 step "merge_$1" $PY merge.py --model "$1"
}

for phase in "$@"; do
  case $phase in
    gate)
      merged $FINAL 0 || exit 1
      step gate bash fit.sh /workspace/models/$FINAL 0 /workspace/out/internal-readout-gap/own_lens_gate --n_prompts 3 || exit 1
      step gate_load $PY -c "
import glob, jlens
lens = jlens.JacobianLens.load(glob.glob('/workspace/out/internal-readout-gap/own_lens_gate/*_jacobian_lens.pt')[0])
print('the gate lens loads:', lens)" || exit 1 ;;
    fetch)
      step fetch_reads $PY fetch_reads.py --models gemma-3-27b-it $FINAL gemma-distilled-step48 --words_only qwen3.6-27b \
        gemma-distilled-step24 gemma-distilled-step72 || exit 1 ;;
    fits)
      f0() { fit $BASE 0 gemma-3-27b-it; }; f1() { fit /workspace/models/$FINAL 1 $FINAL; }
      side_by_side f0 f1 || exit 1 ;;
    early)   # the early reproduction check, at 20 and at 50 prompts of the control fit (CPU; the GPUs are fitting)
      export CUDA_VISIBLE_DEVICES=
      C=$LENSES/gemma-3-27b-it; D=$LENSES/$FINAL
      step reread_published $PY reread.py --model gemma-3-27b-it --lens published --name published-reread --limit 20 || exit 1
      step check_reread $PY compare_lenses.py words --a gemma-3-27b-it.published-reread --b gemma-3-27b-it || exit 1
      for n in 20 50; do
        until snap=$($PY snapshots.py --fit_dir "$C" --find $n); do sleep 30; done
        step curve_at_$n $PY compare_lenses.py curve --ours "$C/gemma-3-27b-it_convergence.csv" || exit 1
        step curve_final_vs_control_at_$n $PY compare_lenses.py curve --ours "$D/${FINAL}_convergence.csv"           --other "$C/gemma-3-27b-it_convergence.csv" || exit 1
        step reread_control_at_$n $PY reread.py --model gemma-3-27b-it --lens "$snap" --name control-at-$n --limit 20 || exit 1
        step words_control_at_$n $PY compare_lenses.py words --a gemma-3-27b-it.control-at-$n --b gemma-3-27b-it || exit 1
      done ;;
    rescore)  # step 0 on the box: rate_gap.py again on Gigi's updated hero9_swap.csv (it needs every model's activations)
      step fetch_all_reads $PY fetch_reads.py --models gemma-3-27b-it qwen3.6-27b gemma-distilled-step24         gemma-distilled-step48 gemma-distilled-step72 $FINAL || exit 1
      ( cd ../distillation_similarity && CUDA_VISIBLE_DEVICES= step rate_gap_gigi_corrections $PY rate_gap.py ) || exit 1 ;;
    fit_final)      # $FINAL's lens alone, on GPU 0
      fit /workspace/models/$FINAL 0 $FINAL || exit 1 ;;
    watch_final)    # the watchers alone, for a fit of $FINAL that is already running
      watch $FINAL ;;
    upload_final)   # the finished lens to HF, with the size check
      step upload_$FINAL $PY upload_lens.py --fit $FINAL || exit 1 ;;
    fit48)
      merged gemma-distilled-step48 "${GPU48:?set GPU48}" || exit 1
      fit /workspace/models/gemma-distilled-step48 "$GPU48" gemma-distilled-step48 || exit 1 ;;
    *) echo "unknown phase $phase" >> "$STATUS"; exit 1 ;;
  esac
  echo "$(date -u +%FT%TZ) phase $phase done" >> "$STATUS"
done
