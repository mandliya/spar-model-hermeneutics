#!/usr/bin/env bash
# hero7 (8 Nov 2026, night; GG: "find another control example ... so we have it to compare"): Llama 3.3 70B (fp16, CoreWeave)
# on ALL 29 tasks under the hard end rule plus the CONTROL worked example (agent.hack=worked_example_control,
# agent-interp-envs/src/agent_interp_envs/hacks.py, patches/end_task_required.patch): the same six-line example as
# worked_example (hero6), same file name and test command, but the demonstrated run ends with "2 passed, 1 failed" and the
# report says so. If Llama's bad acts under the success example survive this one, they are Llama's; if they vanish, the
# "all tests pass" line was teaching them. Same configs, provider pin, step caps and run counts as hero4/hero6, so the
# three compare row by row. Stages: hero7_control (7 main tasks), hero7_control_key (ai_rd, capped agent key),
# hero7_control_all (the other 20 + swe_rebench_v2 via scripts/swe_modal.py). Estimate $60-100 incl. grading, cap $400
# (the shared watchdog scripts/spend_watchdog.sh hero5 is already running). Refuses to launch below MIN_CREDITS_USD.
# Detached, absolute paths; the laptop only has to stay awake for the ~8 minutes of launching:
#   nohup caffeinate -i bash /abs/path/gigi/3-forensics/scripts/hero7_launch.sh > /abs/log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"; ROOT="$(cd ../.. && pwd)"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
OV="agent.end_task_required=true,agent.end_task_reprompts=10,agent.hack=worked_example_control"
L="$HERE/results/logs/hero7"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/launch.txt"; }
bal=$(FORENSICS_ENV_FILE="$ROOT/.env" python3 "$HERE/scripts/hero_usage.py" 2>/dev/null | grep -i "org credits" | grep -oE '\$[0-9][0-9,]*\.[0-9]+ remaining' | grep -oE '[0-9][0-9,]*\.[0-9]+' | tr -d ,)
say "org credit balance: ${bal:-unknown}"
python3 -c "import sys; sys.exit(0 if float('${bal:-0}') >= float('${MIN_CREDITS_USD:-300}') else 1)" || { say "balance below ${MIN_CREDITS_USD:-300} or unknown: NOT launching"; exit 1; }
MAIN="coding:20 coding_norules:20 precommit_hook:20 dashboard_costly_failure:10 grade_your_own:5 optimize_rendering_engine:5 optimize_rendering_engine_graded:5"
OTHER="async_conversion:5 chess:19 copying_of_answers:10 currency_conversion:20 eval_tampering:21 funding_email:20 lazy_investigation:14 leaked_scorer:10 leaked_tests:5 lucky_seed:9 lucky_seed_readable_grader:8 norvane:20 optimize_lru_cache:5 parse_duration:21 puppeteer:5 reference_extraction:10 revert_or_refactor:5 secret_number:20 tictactoe:19 tower_of_hanoi:20"
say "launching llama-3.3-70b, hard rule + control example (stage hero7_control): $MAIN"
EXTRA_OV="$OV" MAXC=0 GAP=${GAP:-12} bash "$HERE/scripts/modal_full.sh" hero7_control llama-3.3-70b $MAIN
say "launching ai_rd:10 with the capped agent key (stage hero7_control_key)"
EXTRA_OV="$OV" MAXC=0 GAP=${GAP:-12} MODAL_SECRETS=openrouter-agent bash "$HERE/scripts/modal_full.sh" hero7_control_key llama-3.3-70b ai_rd:10
say "launching the other 20 (stage hero7_control_all): $OTHER + swe_rebench_v2:10"
EXTRA_OV="$OV" MAXC=0 GAP=${GAP:-12} bash "$HERE/scripts/modal_full.sh" hero7_control_all llama-3.3-70b $OTHER
mkdir -p "$HERE/results/logs/hero7_control_all"
( cd "$HERE/agent-interp-envs" && nohup env -u VIRTUAL_ENV uv run modal run --detach ../scripts/swe_modal.py \
    --config ../configs/swe_rebench_v2__llama-3.3-70b.yaml --count 10 --stage hero7_control_all --overrides "$OV" \
    > "$HERE/results/logs/hero7_control_all/swe_rebench_v2__llama-3.3-70b.log" 2>&1 & )
say "launched (hero7)"
