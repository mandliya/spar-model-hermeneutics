#!/usr/bin/env bash
# Llama's last try, the other 21 environments (7 Nov 2026, GG: "run it on everything"): the hard end rule
# (agent.end_task_required=true, patches/end_task_required.patch; the environment's own finish tool where it has one,
# else a synthetic end_task) on the 21 environments outside the 8 main tasks, stage hero4_llama_all. Same configs,
# provider pin (Llama 3.3 70B fp16 on CoreWeave, no fallback), step caps and run counts as Llama's soft-check pass on
# the same environments (scripts/hero2_launch_other_envs.sh, stage hero2_llama_all), so the two passes compare row by
# row. swe_rebench_v2 goes through scripts/swe_modal.py with the same overrides (main key, redacted on fetch).
# Estimate ~$45 (tokens + grading), approved cap $150. Refuses to launch if the org balance is below MIN_CREDITS_USD.
# Detached, absolute paths; the laptop only has to stay awake for the ~5 minutes of launching:
#   nohup caffeinate -i bash /abs/path/gigi/3-forensics/scripts/hero4_launch_llama_all.sh > /abs/log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"; ROOT="$(cd ../.. && pwd)"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
OV="agent.end_task_required=true,agent.end_task_reprompts=10"
L="$HERE/results/logs/hero4"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/launch_all.txt"; }
bal=$(FORENSICS_ENV_FILE="$ROOT/.env" python3 "$HERE/scripts/hero_usage.py" 2>/dev/null | grep -i "org credits" | grep -oE '\$[0-9][0-9,]*\.[0-9]+ remaining' | grep -oE '[0-9][0-9,]*\.[0-9]+' | tr -d ,)
say "org credit balance: ${bal:-unknown}"
python3 -c "import sys; sys.exit(0 if float('${bal:-0}') >= float('${MIN_CREDITS_USD:-300}') else 1)" || { say "balance below ${MIN_CREDITS_USD:-300} or unknown: NOT launching"; exit 1; }
ENVS="async_conversion:5 chess:19 copying_of_answers:10 currency_conversion:20 eval_tampering:21 funding_email:20 lazy_investigation:14 leaked_scorer:10 leaked_tests:5 lucky_seed:9 lucky_seed_readable_grader:8 norvane:20 optimize_lru_cache:5 parse_duration:21 puppeteer:5 reference_extraction:10 revert_or_refactor:5 secret_number:20 tictactoe:19 tower_of_hanoi:20"
say "launching llama-3.3-70b with the hard end rule (stage hero4_llama_all): $ENVS + swe_rebench_v2:10"
EXTRA_OV="$OV" MAXC=0 GAP=${GAP:-12} bash "$HERE/scripts/modal_full.sh" hero4_llama_all llama-3.3-70b $ENVS
mkdir -p "$HERE/results/logs/hero4_llama_all"
( cd "$HERE/agent-interp-envs" && nohup env -u VIRTUAL_ENV uv run modal run --detach ../scripts/swe_modal.py \
    --config ../configs/swe_rebench_v2__llama-3.3-70b.yaml --count 10 --stage hero4_llama_all --overrides "$OV" \
    > "$HERE/results/logs/hero4_llama_all/swe_rebench_v2__llama-3.3-70b.log" 2>&1 & )
say "launched (hero4_llama_all)"
