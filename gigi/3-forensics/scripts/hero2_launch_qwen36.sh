#!/usr/bin/env bash
# Qwen3.6 27B on its own (7 Nov 2026, after the 6 Nov call: nobody had run Qwen3.6 27B solo, and John and Yuqi worry it
# might be too dumb), fp8 DeepInfra pinned, no fallback, with the end-of-run check (agent.end_confirm=true), the same
# step caps, timeouts and run counts as the other hero2 passes. Four stages, all launched at once, detached on Modal:
#   hero2_qwen36        the 7 gap environments, thinking OFF (configs/<env>__qwen3.6-27b.yaml)
#   hero2_qwen36_key    ai_rd:10 with the capped agent key (Modal secret openrouter-agent-gigi), thinking OFF
#   hero2_qwen36_all    the other 21 environments, thinking OFF (swe_rebench_v2 through scripts/swe_modal.py)
#   hero2_qwen36_think  the four cheating environments x10, thinking ON (configs/<env>__qwen3.6-27b-think.yaml)
# Budget (the user, 7 Nov): up to $500 for the solo run and $100 for the thinking run; scripts/spend_watchdog.sh stops
# the qwen3.6 apps at CAP_USD. Refuses to launch if the org credit balance is below MIN_CREDITS_USD (default 300).
#   nohup caffeinate -i bash /abs/path/gigi/3-forensics/scripts/hero2_launch_qwen36.sh > /abs/log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"; ROOT="$(cd ../.. && pwd)"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
L="$HERE/results/logs/hero2"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/launch_qwen36.txt"; }
bal=$(FORENSICS_ENV_FILE="$ROOT/.env" python3 "$HERE/scripts/hero_usage.py" 2>/dev/null | grep -i "org credits" | grep -oE '\$[0-9][0-9,]*\.[0-9]+ remaining' | grep -oE '[0-9][0-9,]*\.[0-9]+' | tr -d ,)
say "org credit balance: ${bal:-unknown}"
python3 -c "import sys; sys.exit(0 if float('${bal:-0}') >= float('${MIN_CREDITS_USD:-300}') else 1)" || { say "balance below ${MIN_CREDITS_USD:-300} or unknown: NOT launching"; exit 1; }
GAP7="coding:20 coding_norules:20 precommit_hook:20 dashboard_costly_failure:10 grade_your_own:5 optimize_rendering_engine:5 optimize_rendering_engine_graded:5"
OTHER="async_conversion:5 chess:20 copying_of_answers:10 currency_conversion:20 eval_tampering:20 funding_email:20 lazy_investigation:20 leaked_scorer:10 leaked_tests:5 lucky_seed:10 lucky_seed_readable_grader:10 norvane:20 optimize_lru_cache:10 parse_duration:20 puppeteer:5 reference_extraction:10 revert_or_refactor:5 secret_number:20 tictactoe:20 tower_of_hanoi:20"
THINK="grade_your_own:10 currency_conversion:10 lazy_investigation:10 eval_tampering:10"
export EXTRA_OV="agent.end_confirm=true" MAXC=0 GAP=${GAP:-12}
say "launching hero2_qwen36 (qwen3.6-27b): $GAP7"
bash "$HERE/scripts/modal_full.sh" hero2_qwen36 qwen3.6-27b $GAP7
say "launching hero2_qwen36_key (qwen3.6-27b): ai_rd:10"
MODAL_SECRETS=openrouter-agent bash "$HERE/scripts/modal_full.sh" hero2_qwen36_key qwen3.6-27b ai_rd:10
say "launching hero2_qwen36_think (qwen3.6-27b-think): $THINK"
bash "$HERE/scripts/modal_full.sh" hero2_qwen36_think qwen3.6-27b-think $THINK
say "launching hero2_qwen36_all (qwen3.6-27b): $OTHER + swe_rebench_v2:10"
bash "$HERE/scripts/modal_full.sh" hero2_qwen36_all qwen3.6-27b $OTHER
mkdir -p "$HERE/results/logs/hero2_qwen36_all"
( cd "$HERE/agent-interp-envs" && nohup env -u VIRTUAL_ENV uv run modal run --detach ../scripts/swe_modal.py \
    --config ../configs/swe_rebench_v2__qwen3.6-27b.yaml --count 10 --stage hero2_qwen36_all --overrides agent.end_confirm=true \
    > "$HERE/results/logs/hero2_qwen36_all/swe_rebench_v2__qwen3.6-27b.log" 2>&1 & )
say "all four stages launched"
