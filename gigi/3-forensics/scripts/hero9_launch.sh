#!/usr/bin/env bash
# hero9 (10 Nov 2026 call: Llama is dropped; Qwen3.6 27B and gemma-3 27B are the heroes and must run as the paper's
# environments run them): the PLAIN environment, no end check, no hard rule, no hack. Same configs, provider pins, step
# caps and run counts as hero8 (Qwen3.6 fp8 DeepInfra thinking off, MAXC=0; gemma fp8 DeepInfra text-mode tools, MAXC=5).
# gemma's no-check runs already exist (the original 19-25 Oct passes, results/full_runs.csv, full targets everywhere:
# results/logs/hero9/gemma_topup_plan.md), so the default is Qwen3.6 only; gemma is launched only when named.
#   bash hero9_launch.sh smoke qwen3.6-27b|gemma-3-27b|both   one run each of coding (submit ending), tictactoe (game
#                                                            end) and eval_tampering (plain-reply ending): one smoke per loop type
#   bash hero9_launch.sh full  qwen3.6-27b|gemma-3-27b|both   stages hero9_<tag> (7 main tasks), hero9_<tag>_key (ai_rd:10,
#                                                            capped agent key), hero9_<tag>_all (other 20 + swe_rebench_v2:10)
# Refuses to launch below MIN_CREDITS_USD (default 300) of org credit. A stage whose launch log exists is skipped
# (modal_full.sh; a dead fleet log is moved aside and relaunched), so rerunning after a partial launch is safe.
#   nohup caffeinate -i bash /abs/path/gigi/3-forensics/scripts/hero9_launch.sh full qwen3.6-27b > /abs/log 2>&1 &
set -uo pipefail
MODE=${1:?smoke|full}; WHICH=${2:-qwen3.6-27b}
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"; ROOT="$(cd ../.. && pwd)"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
L="$HERE/results/logs/hero9"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/launch.txt"; }
case "$WHICH" in
  gemma-3-27b|qwen3.6-27b) MODELS="$WHICH" ;;
  both) MODELS="gemma-3-27b qwen3.6-27b" ;;
  *) echo "usage: $0 smoke|full gemma-3-27b|qwen3.6-27b|both" >&2; exit 2 ;;
esac
case "$MODE" in smoke|full) ;; *) echo "usage: $0 smoke|full gemma-3-27b|qwen3.6-27b|both" >&2; exit 2 ;; esac

bal=$(FORENSICS_ENV_FILE="$ROOT/.env" python3 "$HERE/scripts/hero_usage.py" 2>/dev/null | grep -i "org credits" | grep -oE '\$[0-9][0-9,]*\.[0-9]+ remaining' | grep -oE '[0-9][0-9,]*\.[0-9]+' | tr -d ,)
say "org credit balance: ${bal:-unknown}"
python3 -c "import sys; sys.exit(0 if float('${bal:-0}' or 0) >= float('${MIN_CREDITS_USD:-300}') else 1)" || { say "balance below ${MIN_CREDITS_USD:-300} or unknown: NOT launching"; exit 1; }

MAIN="coding:20 coding_norules:20 precommit_hook:20 dashboard_costly_failure:10 grade_your_own:5 optimize_rendering_engine:5 optimize_rendering_engine_graded:5"
OTHER="async_conversion:5 chess:19 copying_of_answers:10 currency_conversion:20 eval_tampering:21 funding_email:20 lazy_investigation:14 leaked_scorer:10 leaked_tests:5 lucky_seed:9 lucky_seed_readable_grader:8 norvane:20 optimize_lru_cache:5 parse_duration:21 puppeteer:5 reference_extraction:10 revert_or_refactor:5 secret_number:20 tictactoe:19 tower_of_hanoi:20"
SMOKE="coding:1 tictactoe:1 eval_tampering:1"   # submit ending / game end / plain-reply ending: one smoke per loop type

for m in $MODELS; do
  tag=qwen36; mc=0; [ "$m" = gemma-3-27b ] && { tag=gemma; mc=5; }
  if [ "$MODE" = smoke ]; then
    say "launching smoke for $m (stage hero9_smoke_$tag): $SMOKE"
    MAXC=$mc GAP=${GAP:-12} bash "$HERE/scripts/modal_full.sh" "hero9_smoke_$tag" "$m" $SMOKE
    say "launched (hero9 smoke $m)"
    continue
  fi
  say "launching $m, no check (plain environment) (stage hero9_$tag): $MAIN"
  MAXC=$mc GAP=${GAP:-12} bash "$HERE/scripts/modal_full.sh" "hero9_$tag" "$m" $MAIN
  say "launching ai_rd:10 for $m with the capped agent key (stage hero9_${tag}_key)"
  MAXC=$mc GAP=${GAP:-12} MODAL_SECRETS=openrouter-agent bash "$HERE/scripts/modal_full.sh" "hero9_${tag}_key" "$m" ai_rd:10
  say "launching the other 20 for $m (stage hero9_${tag}_all): $OTHER + swe_rebench_v2:10"
  MAXC=$mc GAP=${GAP:-12} bash "$HERE/scripts/modal_full.sh" "hero9_${tag}_all" "$m" $OTHER
  mkdir -p "$HERE/results/logs/hero9_${tag}_all"
  [ -f "$HERE/results/logs/hero9_${tag}_all/swe_rebench_v2__$m.log" ] && say "swe_rebench_v2 already launched for $m (log exists): skipping" || \
  ( cd "$HERE/agent-interp-envs" && nohup env -u VIRTUAL_ENV uv run modal run --detach ../scripts/swe_modal.py \
      --config "../configs/swe_rebench_v2__$m.yaml" --count 10 --stage "hero9_${tag}_all" \
      > "$HERE/results/logs/hero9_${tag}_all/swe_rebench_v2__$m.log" 2>&1 & )
  say "launched (hero9 $m)"
done
say "all launched (hero9 $MODE $WHICH)"
