#!/usr/bin/env bash
# Gather what the own-lens run produced off the box (logs, result files, figures) into
# one folder and upload it to internal-readout-gap/own_lens/local_analysis on HF, with the completeness check.
#   bash assemble_local_analysis.sh
# Safe to rerun: it copies again and uploads again; nothing on HF is deleted. Prints the revision, which the
# report's links to these logs must use.
set -euo pipefail
E=$(cd "$(dirname "$0")/.." && pwd)
S=$E/../../dyl-probe-repro/scratch/internal-readout-gap
A=$S/own_lens_local_analysis
mkdir -p "$A/logs" "$A/results" "$A/figures"
cp "$S"/logs_gigi_corrections/* "$S"/logs_readout_diffing/* "$S"/logs_positions/* "$A/logs/"
[ -f "$S/logs_local_analysis/audit_app_build.log" ] && cp "$S/logs_local_analysis/audit_app_build.log" "$A/logs/"
cp "$S/fetched/distillation_similarity/similarity.json" "$A/results/similarity_gigi_corrections.json"
cp "$S/fetched/distillation_similarity/where_it_moved.json" "$A/results/where_it_moved_gigi_corrections.json"
cp "$S"/positions_gap_gemma-3-27b-it__qwen3.6-27b.json "$A/results/"
D=$S/own_lens_run/distillation_similarity/outputs/distributions
cp "$D"/similarity_methods*.json "$A/results/"
for f in robust_intervals.json yardstick_ties.json nla_root_cause.json similarity_methods_hybrids.json; do [ -f "$D/$f" ] && cp "$D/$f" "$A/results/"; done
cp "$E"/report/own_lens_by_depth.png "$E"/report/positions_distillation.png "$E"/report/positions_lie_detector.png "$E"/report/scoring_methods.png "$E"/report/nla_root_cause.png "$A/figures/"
cd "$E/own_lens"
set -a; . ../../../dyl-probe-repro/.env; set +a
export HF_HOME=$HOME/.cache/huggingface HF_HUB_DISABLE_XET=1
~/.venvs/dyl-probe-repro/bin/python upload_local.py --folder "$A" 2>&1 | tail -2
