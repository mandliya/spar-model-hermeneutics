#!/usr/bin/env bash
# Qwen3-Next-80B and gemma-3-27b on the other 21 environments with the end-of-run confirmation on (agent.end_confirm=true),
# stages hero2_qwen_all and hero2_gemma_all (GG approved 6 Nov 2026, budget $300). Same configs, provider pins (DeepInfra
# fp8, no fallback), step caps and timeouts as their original full runs; run counts = the original full runs' nominal
# counts. swe_rebench_v2 goes through scripts/swe_modal.py as for Llama's hero2_llama_all (main key, redacted on fetch).
# Refuses to launch if the org credit balance is below MIN_CREDITS_USD (default 300), so a running fleet is never cut.
# Detached, absolute paths:
#   nohup caffeinate -i bash /abs/path/gigi/3-forensics/scripts/hero2_launch_other_envs_qg.sh > /abs/log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"; ROOT="$(cd ../.. && pwd)"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
L="$HERE/results/logs/hero2"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/launch_other_envs_qg.txt"; }
bal=$(FORENSICS_ENV_FILE="$ROOT/.env" python3 "$HERE/scripts/hero_usage.py" 2>/dev/null | grep -i "org credits" | grep -oE '\$[0-9][0-9,]*\.[0-9]+ remaining' | grep -oE '[0-9][0-9,]*\.[0-9]+' | tr -d ,)
say "org credit balance: ${bal:-unknown}"
python3 -c "import sys; sys.exit(0 if float('${bal:-0}') >= float('${MIN_CREDITS_USD:-300}') else 1)" || { say "balance below ${MIN_CREDITS_USD:-300} or unknown: NOT launching"; exit 1; }
ENVS="async_conversion:5 chess:20 copying_of_answers:10 currency_conversion:20 eval_tampering:20 funding_email:20 lazy_investigation:20 leaked_scorer:10 leaked_tests:5 lucky_seed:10 lucky_seed_readable_grader:10 norvane:20 optimize_lru_cache:10 parse_duration:20 puppeteer:5 reference_extraction:10 revert_or_refactor:5 secret_number:20 tictactoe:20 tower_of_hanoi:20"
for m in qwen3-next-80b gemma-3-27b; do
  tag=qwen; mc=0; [ "$m" = gemma-3-27b ] && { tag=gemma; mc=5; }   # gemma: 5 at a time per environment (DeepInfra 429s in the smoke run)
  say "launching $m (stage hero2_${tag}_all): $ENVS"
  EXTRA_OV="agent.end_confirm=true" MAXC=$mc GAP=12 bash "$HERE/scripts/modal_full.sh" "hero2_${tag}_all" "$m" $ENVS
  mkdir -p "$HERE/results/logs/hero2_${tag}_all"
  ( cd "$HERE/agent-interp-envs" && nohup env -u VIRTUAL_ENV uv run modal run --detach ../scripts/swe_modal.py \
      --config ../configs/swe_rebench_v2__$m.yaml --count 10 --stage "hero2_${tag}_all" --overrides agent.end_confirm=true \
      > "$HERE/results/logs/hero2_${tag}_all/swe_rebench_v2__$m.log" 2>&1 & )
  say "launched $m"
done
say "both models launched (hero2_qwen_all, hero2_gemma_all)"
