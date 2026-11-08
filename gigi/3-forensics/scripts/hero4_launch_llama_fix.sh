#!/usr/bin/env bash
# Rerun of the four environments whose own loop ignored the synthetic end_task before the 8 Nov 2026 fix (chess, tictactoe,
# coding, coding_norules; plus swe_rebench_v2, whose end_session is now the named finish tool): Llama 3.3 70B fp16 on
# CoreWeave under the hard end rule (agent.end_task_required=true, patches/end_task_required.patch), same counts as the
# soft-check and pre-fix passes so the rows compare one to one. Stage hero4_llama_fix. Estimate ~$40, cap $150 (the shared
# watchdog scripts/spend_watchdog.sh hero5 covers all of today's Llama runs). Refuses to launch below MIN_CREDITS_USD.
#   nohup caffeinate -i bash /abs/path/gigi/3-forensics/scripts/hero4_launch_llama_fix.sh > /abs/log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"; ROOT="$(cd ../.. && pwd)"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
OV="agent.end_task_required=true,agent.end_task_reprompts=10"
L="$HERE/results/logs/hero4"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/launch_fix.txt"; }
bal=$(FORENSICS_ENV_FILE="$ROOT/.env" python3 "$HERE/scripts/hero_usage.py" 2>/dev/null | grep -i "org credits" | grep -oE '\$[0-9][0-9,]*\.[0-9]+ remaining' | grep -oE '[0-9][0-9,]*\.[0-9]+' | tr -d ,)
say "org credit balance: ${bal:-unknown}"
python3 -c "import sys; sys.exit(0 if float('${bal:-0}') >= float('${MIN_CREDITS_USD:-300}') else 1)" || { say "balance below ${MIN_CREDITS_USD:-300} or unknown: NOT launching"; exit 1; }
ENVS="chess:19 tictactoe:19 coding:20 coding_norules:20"
say "launching llama-3.3-70b with the hard end rule after the end_task fix (stage hero4_llama_fix): $ENVS + swe_rebench_v2:10"
EXTRA_OV="$OV" MAXC=0 GAP=${GAP:-12} bash "$HERE/scripts/modal_full.sh" hero4_llama_fix llama-3.3-70b $ENVS
mkdir -p "$HERE/results/logs/hero4_llama_fix"
( cd "$HERE/agent-interp-envs" && nohup env -u VIRTUAL_ENV uv run modal run --detach ../scripts/swe_modal.py \
    --config ../configs/swe_rebench_v2__llama-3.3-70b.yaml --count 10 --stage hero4_llama_fix --overrides "$OV" \
    > "$HERE/results/logs/hero4_llama_fix/swe_rebench_v2__llama-3.3-70b.log" 2>&1 & )
say "launched (hero4_llama_fix)"
