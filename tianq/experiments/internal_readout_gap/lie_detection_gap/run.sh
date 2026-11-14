#!/usr/bin/env bash
# Question 2 on the box, detached and resumable (PLAN.md, stage 2). Run one or more phases, in this order:
#   (nohup setsid bash run.sh reads > /dev/null 2>&1 < /dev/null &)       # activations and J-lens words, every read
#   (nohup setsid bash run.sh nla upload > /dev/null 2>&1 < /dev/null &)   # NLA sentences, then the upload
# A "read" is <target>__read_by_<model>: the target's rows read by the target itself or by a substitute. The gap
# itself is computed afterwards from the uploaded outputs (gap.py, on a machine with dyl_repro's environment).
# Steps, logs and the status file: ../shared/steps.sh.
set -uo pipefail
cd "$(dirname "$0")"
E=lie_detection_gap
. ../shared/steps.sh
G=gemma-3-27b-it; Q=qwen3-32b
# target:model for every read, split over the two GPUs. The first three are the pairs with an NLA on both sides.
GPU0="$G:$G $G:qwen3.6-27b $Q:$G"
GPU1="$G:gemma-3-12b-it $Q:$Q"
NLA0="$G:$G"
NLA1="$G:gemma-3-12b-it $G:qwen3.6-27b"

reads() {  # reads <gpu> <target:model ...>
  local items=() r
  for r in $2; do items+=("read_${r%%:*}__read_by_${r#*:}:$PY read_rows.py --target ${r%%:*} --model ${r#*:}"); done
  queue "$1" "${items[@]}"
}
nlas() {   # nlas <gpu> <target:model ...>
  local items=() r
  for r in $2; do
    items+=("nla_${r%%:*}__read_by_${r#*:}:$(nla_python "${r#*:}") ../shared/nla_read.py --experiment $E --model ${r#*:} --read ${r%%:*}__read_by_${r#*:}")
  done
  queue "$1" "${items[@]}"
}

for phase in "$@"; do
  case $phase in
    reads)
      r0() { reads 0 "$GPU0"; }; r1() { reads 1 "$GPU1"; }
      side_by_side r0 r1 || exit 1 ;;
    nla)
      n0() { nlas 0 "$NLA0"; }; n1() { nlas 1 "$NLA1"; }
      side_by_side n0 n1 || exit 1 ;;
    upload)
      step nla_check $PY ../shared/nla_check.py --experiment $E || exit 1
      step bundle $PY bundle.py || exit 1
      step upload $PY ../shared/upload.py --experiment $E || exit 1 ;;
    *) echo "unknown phase $phase" >> "$STATUS"; exit 1 ;;
  esac
  echo "$(date -u +%FT%TZ) phase $phase done" >> "$STATUS"
done
