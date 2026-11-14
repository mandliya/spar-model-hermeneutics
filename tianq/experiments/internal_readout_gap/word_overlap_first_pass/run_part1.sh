#!/usr/bin/env bash
# ARCHIVED (8 Nov 2026 first pass). Kept only so that experiment can be reproduced; do not import or extend.
# New runs are described in ../PLAN.md.
# Part 1 of PLAN.md on box A, detached and resumable: Qwen3-Next-80B (target) vs gemma-3-27b-it (substitute).
#   (nohup setsid bash tianq/experiments/internal_readout_gap/word_overlap_first_pass/run_part1.sh > /dev/null 2>&1 &)
# Each stage skips work already on disk, so rerunning picks up where it stopped. One line per stage start/end in
# /workspace/logs/part1_status.txt; each stage's output in /workspace/logs/part1_<stage>.log.
set -uo pipefail
cd "$(dirname "$0")"
set -a; . /workspace/.env; set +a
export HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venvs/internal-readout-gap/bin/python
NLA_PY=/workspace/venvs/nla/bin/python   # the vendored NLA client expects transformers 4.x
OUT=${READOUT_OUT:-/workspace/out/internal-readout-gap}
STATUS=/workspace/logs/part1_status.txt
QWEN_PROMPTS=${QWEN_PROMPTS:-200}
GEMMA_LENS=$(ls /workspace/hf/hub/models--neuronpedia--jacobian-lens/snapshots/*/gemma-3-27b-it/jlens/Salesforce-wikitext/gemma-3-27b-it_jacobian_lens.pt)
QWEN_LENS=$OUT/lenses/qwen3-next-80b_jacobian_lens_n$QWEN_PROMPTS.pt

stage() {   # stage <name> <command...>
  local name=$1; shift
  echo "$(date -u +%FT%TZ) start $name" >> "$STATUS"
  "$@" >> "/workspace/logs/part1_$name.log" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $name exit=$rc" >> "$STATUS"
  return $rc
}

stage gemma_readout $PY readout.py --model gemma-3-27b-it --lens "$GEMMA_LENS" || exit 1
stage gemma_nla $NLA_PY nla.py --model gemma-3-27b-it || exit 1
[ -s "$QWEN_LENS" ] || stage qwen_lens_fit $PY lens_fit.py --model qwen3-next-80b --prompts "$QWEN_PROMPTS" || exit 1
stage qwen_readout $PY readout.py --model qwen3-next-80b --lens "$QWEN_LENS" || exit 1
stage metrics $PY metrics.py --target qwen3-next-80b --substitute gemma-3-27b-it || exit 1
stage upload $PY upload.py --part part1 || exit 1
echo "$(date -u +%FT%TZ) part1 done" >> "$STATUS"
