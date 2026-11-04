#!/usr/bin/env bash
# After the hero2 gap-environment runs (scripts/hero2_launch_solo.sh) have all finished on Modal, launch Llama's second
# pass on the other 21 environments with the confirmation on (stage hero2_llama_all) — but only if the OpenRouter org
# credit balance is still above MIN_CREDITS_USD (default 150), so a running fleet is never cut by a 403. Detached:
#   nohup caffeinate -i bash gigi/3-forensics/scripts/hero2_launch_other_envs.sh &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"; ROOT="$(cd ../.. && pwd)"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
L=results/logs/hero2; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/launch_other_envs.txt"; }
say "waiting for the hero2 gap-environment launches to finish"
until grep -q "all three models launched" "$L/launch_solo.txt" 2>/dev/null; do sleep 60; done
for i in $(seq 1 600); do   # up to 10 h: wait until none of the hero2 apps is still running
  ids=$(grep -ohE "ap-[A-Za-z0-9]+" results/logs/hero2_*/*.log 2>/dev/null | sort -u)
  running=$(cd agent-interp-envs && env -u VIRTUAL_ENV uv run modal app list --json 2>/dev/null | python3 -c "
import sys, json; ids = set(sys.argv[1].split()); apps = json.load(sys.stdin)
print(sum(1 for a in apps if (a.get('App ID') or a.get('app_id')) in ids and 'stopped' not in str(a.get('State', a.get('state', ''))).lower()))" "$ids")
  [ "${running:-1}" = 0 ] && break
  [ $((i % 30)) = 1 ] && say "waiting: $running hero2 apps still running"; sleep 60
done
bal=$(FORENSICS_ENV_FILE="$ROOT/.env" python3 scripts/hero_usage.py 2>/dev/null | grep -iE "credits.*remaining|remaining" | grep -oE '[0-9][0-9,]*\.[0-9]+' | head -1 | tr -d ,)
say "org credit balance now: ${bal:-unknown}"
python3 -c "import sys; sys.exit(0 if float('${bal:-0}') >= float('${MIN_CREDITS_USD:-150}') else 1)" || { say "balance below ${MIN_CREDITS_USD:-150}: NOT launching the other environments (ask John to top up, then run: EXTRA_OV=agent.end_confirm=true MAXC=0 bash scripts/modal_full.sh hero2_llama_all llama-3.3-70b <env:n ...>)"; exit 0; }
EXTRA_OV="agent.end_confirm=true" MAXC=0 GAP=12 bash scripts/modal_full.sh hero2_llama_all llama-3.3-70b async_conversion:5 chess:19 copying_of_answers:10 currency_conversion:20 eval_tampering:21 funding_email:20 lazy_investigation:14 leaked_scorer:10 leaked_tests:5 lucky_seed:9 lucky_seed_readable_grader:8 norvane:20 optimize_lru_cache:5 parse_duration:21 puppeteer:5 reference_extraction:10 revert_or_refactor:5 secret_number:20 tictactoe:19 tower_of_hanoi:20
( cd agent-interp-envs && nohup env -u VIRTUAL_ENV uv run modal run --detach ../scripts/swe_modal.py --config ../configs/swe_rebench_v2__llama-3.3-70b.yaml --count 10 --stage hero2_llama_all > ../results/logs/hero2_llama_all/swe_rebench_v2__llama-3.3-70b.log 2>&1 & )
say "other environments launched (hero2_llama_all)"
