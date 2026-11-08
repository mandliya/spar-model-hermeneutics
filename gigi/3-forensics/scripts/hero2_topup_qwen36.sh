#!/usr/bin/env bash
# One top-up of the Qwen3.6 27B solo cells that lost runs to passing provider errors (rate limits, empty responses;
# 8 Nov 2026, from the three loss audits). Context-window losses are not rerun: DeepInfra's Qwen3.6 endpoint has an
# 81,920-token context and those tasks need more, so a rerun would fail the same way. Three new stages, same settings
# as scripts/hero2_launch_qwen36.sh; then scripts/hero_solo_foldin.sh pools each top-up stage with its original stages
# into the same per-model files. Detached, absolute paths:
#   nohup caffeinate -i bash /abs/.../scripts/hero2_topup_qwen36.sh > /abs/log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
L="$HERE/results/logs/hero2"; say () { echo "$(date '+%F %T') $*" >> "$L/topup_qwen36.txt"; }
export EXTRA_OV="agent.end_confirm=true" MAXC=0 GAP=12
say "launching top-ups"
bash "$HERE/scripts/modal_full.sh" hero2_qwen36_topup qwen3.6-27b coding_norules:3 dashboard_costly_failure:2
bash "$HERE/scripts/modal_full.sh" hero2_qwen36_think_topup qwen3.6-27b-think grade_your_own:4
bash "$HERE/scripts/modal_full.sh" hero2_qwen36_all_topup qwen3.6-27b async_conversion:1 chess:1 currency_conversion:1 leaked_tests:1 norvane:1 reference_extraction:1
say "top-ups launched"
run_one () {   # model, stages, log-dir name, out prefix, audit name
  local m=$1 st=$2 LD="$HERE/results/logs/hero2/$3" pre=$4 aud=$5; mkdir -p "$LD"
  while pgrep -f "scripts/hero_solo_foldin.sh" > /dev/null; do sleep 60; done
  say "$m ($st): fold-in"
  HERO_ALL_ENVS=${6:-0} WAIT_MINUTES=600 CEIL=${CEIL:-3300} GRADE_CAP=100 HERO_MODEL="$m" HERO_STAGES="$st" \
    HERO_OUT_PREFIX="$pre" HERO_AUDIT="$aud" HERO_LOG_DIR="$LD" \
    bash "$HERE/scripts/hero_solo_foldin.sh" > "$LD/foldin_topup.out" 2>&1
  say "$m ($st): exit $?; $(grep -h 'scored:\|STEP FAILED' "$LD/progress.txt" 2>/dev/null | tail -2 | tr '\n' ';')"
}
run_one qwen3.6-27b-think hero2_qwen36_think,hero2_qwen36_think_topup qwen3.6-27b-think hero2_full_runs__qwen3.6-27b-think hero2_loss_audit__qwen3.6-27b-think.md 1
run_one qwen3.6-27b hero2_qwen36,hero2_qwen36_key,hero2_qwen36_topup qwen3.6-27b hero2_full_runs__qwen3.6-27b hero2_loss_audit__qwen3.6-27b.md
run_one qwen3.6-27b hero2_qwen36_all,hero2_qwen36_all_topup qwen3.6-27b__other21 hero2_full_runs__qwen3.6-27b__other21 hero2_loss_audit__qwen3.6-27b__other21.md 1
say "DONE"
