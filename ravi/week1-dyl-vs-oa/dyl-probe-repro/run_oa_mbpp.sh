#!/usr/bin/env bash
# GPU box: extract, then score, OA MBPP-Honeypot with the official DYL probe.
#
#   ./logged.sh 'tmux new -d -s oa ./run_oa_mbpp.sh'
#
# This directory is a copy of tianq/dyl-probe-repro plus dataset oa_mbpp.
# Completions are on-policy at temperature 0. Default follow-ups are the native
# DYL question, the paper's truthful variant (the one that transferred on ASCII),
# and the cheat/falsify wordings. Override with FOLLOWUPS:
#
#   FOLLOWUPS=all ./run_oa_mbpp.sh
#   FOLLOWUPS="dyl dyl_truthful" ./run_oa_mbpp.sh
#
# Smoke test first: ./run.sh extract --dataset oa_mbpp --followup dyl --limit 8
#
# Each step is its own ./run.sh call, so each gets its own raw log on the Hub.
# Safe to kill and rerun: extract skips rows already stored.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "${FOLLOWUPS:-}" == "all" ]]; then
  args=(all)
else
  # shellcheck disable=SC2206
  args=(${FOLLOWUPS:-dyl dyl_truthful dyl_cheat dyl_falsify})
fi

./run.sh extract --dataset oa_mbpp --followup "${args[@]}"
./run.sh evaluate --dataset oa_mbpp --followup "${args[@]}"
