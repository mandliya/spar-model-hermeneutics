#!/usr/bin/env bash
# top up every restart point to its target (clean 20, cheating 60 per arm), with a 4 h sandbox timeout for Qwen3.6's long
# replays, then run the swap fold-in again with a grading cap that fits the estimate.
set -uo pipefail
cd /Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
L=results/logs/hero2_swap; say () { echo "$(date '+%F %T') $*" >> "$L/topup.txt"; }
say "top-up start (SWAP_TIMEOUT 14400)"
HERO2=1 SWAP_TIMEOUT=14400 bash scripts/hero_swap.sh --topup --arms llama,qwen36,gemma >> "$L/topup.txt" 2>&1; say "top-up launch exit $?"
CEIL=2400 GRADE_CAP=400 bash scripts/hero2_swap_foldin.sh >> "$L/topup.txt" 2>&1; say "fold-in relaunched (waits for the top-up apps, then grades everything)"
