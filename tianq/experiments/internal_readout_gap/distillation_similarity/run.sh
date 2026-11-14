#!/usr/bin/env bash
# Question 1 on the box, detached and resumable (PLAN.md, stage 1). Run one or more phases, in this order:
#   (nohup setsid bash run.sh reads > /dev/null 2>&1 < /dev/null &)        # activations and J-lens words, all models
#   (nohup setsid bash run.sh nla analysis > /dev/null 2>&1 < /dev/null &)  # NLA sentences, then the numbers and upload
# Each GPU takes three models, one after another. A spot too long for one GPU is read in "read_wide", where each
# model gets both GPUs. Steps, logs and the status file: ../shared/steps.sh.
set -uo pipefail
cd "$(dirname "$0")"
E=distillation_similarity
. ../shared/steps.sh
GPU0="gemma-3-27b-it gemma-distilled-step24 gemma-distilled-step48"
GPU1="qwen3.6-27b gemma-distilled-step72 gemma-distilled-step96"

reads() {  # reads <gpus> <models> <step prefix>
  local items=() m
  for m in $2; do items+=("$3_$m:$PY read_spots.py --model $m"); done
  queue "$1" "${items[@]}"
}
nlas() {   # nlas <gpus> <models>
  local items=() m
  for m in $2; do items+=("nla_$m:$(nla_python "$m") ../shared/nla_read.py --experiment $E --model $m"); done
  queue "$1" "${items[@]}"
}

for phase in "$@"; do
  case $phase in
    reads)
      r0() { reads 0 "$GPU0" read; }; r1() { reads 1 "$GPU1" read; }
      side_by_side r0 r1 || exit 1
      reads 0,1 "$GPU0 $GPU1" read_wide || exit 1 ;;
    nla)
      n0() { nlas 0 "$GPU0"; }; n1() { nlas 1 "$GPU1"; }
      side_by_side n0 n1 || exit 1 ;;
    analysis)
      step similarity $PY similarity.py || exit 1
      step rate_gap $PY rate_gap.py || exit 1
      step nla_check $PY ../shared/nla_check.py --experiment $E || exit 1
      step upload $PY ../shared/upload.py --experiment $E || exit 1 ;;
    *) echo "unknown phase $phase" >> "$STATUS"; exit 1 ;;
  esac
  echo "$(date -u +%FT%TZ) phase $phase done" >> "$STATUS"
done
