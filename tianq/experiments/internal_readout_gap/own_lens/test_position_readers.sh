#!/usr/bin/env bash
# The two position readers on a few inputs, on CPU, into a scratch output folder (the GPUs are fitting).
#   (nohup setsid bash test_position_readers.sh > /dev/null 2>&1 < /dev/null &)
# Logs: /workspace/logs_test/. Nothing here is uploaded.
set -uo pipefail
cd "$(dirname "$0")"
E=test
. ../shared/steps.sh
export READOUT_OUT=/workspace/out_test/internal-readout-gap CUDA_VISIBLE_DEVICES=
mkdir -p /workspace/logs_test
STATUS=/workspace/logs_test/status.txt
t() { local name=$1; shift; echo "$(date -u +%FT%TZ) start $name" >> "$STATUS"; "$@" > "/workspace/logs_test/$name.log" 2>&1; echo "$(date -u +%FT%TZ) end $name exit=$?" >> "$STATUS"; }
GATE=/workspace/out/internal-readout-gap/own_lens_gate/gemma-distilled-step96_jacobian_lens.pt
cd ../lie_detection_gap
t reply_gemma $PY read_reply_positions.py --target gemma-3-27b-it --model gemma-3-27b-it --limit 3
t reply_qwen $PY read_reply_positions.py --target gemma-3-27b-it --model qwen3.6-27b --limit 3
LIE_EXPERIMENT=lie_detection_positions t bundle $PY bundle.py
cd ../distillation_similarity
t positions_final $PY read_positions.py --model gemma-distilled-step96 --own_lens $GATE --limit 3
t positions_base $PY read_positions.py --model gemma-3-27b-it --own_lens $GATE --limit 3
t positions_qwen $PY read_positions.py --model qwen3.6-27b --limit 3
t positions_similarity $PY positions_similarity.py
