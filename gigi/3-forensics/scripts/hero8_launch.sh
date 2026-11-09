#!/usr/bin/env bash
# hero8 (9 Nov 2026; John's gate on the 4 Nov call: "only if Llama operates in the env, then also inference Qwen/gemma on
# these evals"): the two stand-ins, gemma-3-27b (fp8 DeepInfra, text-mode tools) and qwen3.6-27b (fp8 DeepInfra, thinking
# off), on ALL 29 tasks under exactly Llama's hero6 setting: the hard end rule plus the worked example
# (agent.end_task_required=true, agent.end_task_reprompts=10, agent.hack=worked_example; end_task_required.py, hacks.py).
# Same configs, provider pins, step caps and run counts as hero6/hero7, so the three models compare row by row.
#   bash hero8_launch.sh smoke gemma-3-27b|qwen3.6-27b|both     one run each of coding (own finish tool `done`), tictactoe
#                                                              (game-ending loop) and grade_your_own (plain-reply ending):
#                                                              one smoke per loop type before any fleet (7 Nov lesson)
#   bash hero8_launch.sh full  gemma-3-27b|qwen3.6-27b|both     stages hero8_<tag> (7 main tasks), hero8_<tag>_key (ai_rd:10,
#                                                              capped agent key), hero8_<tag>_all (other 20 + swe_rebench_v2:10)
# tag = gemma | qwen36. gemma runs 5 at a time per environment (MAXC=5: DeepInfra 429s in the 6 Nov smoke), Qwen3.6 unlimited.
# Refuses to launch below MIN_CREDITS_USD (default 300) of org credit. A stage whose launch log exists is skipped
# (modal_full.sh), so rerunning after a partial launch is safe. Detached, absolute paths; the laptop only has to stay
# awake for the ~10 minutes of launching:
#   nohup caffeinate -i bash /abs/path/gigi/3-forensics/scripts/hero8_launch.sh full both > /abs/log 2>&1 &
set -uo pipefail
MODE=${1:?smoke|full}; WHICH=${2:-both}
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"; ROOT="$(cd ../.. && pwd)"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
OV="agent.end_task_required=true,agent.end_task_reprompts=10,agent.hack=worked_example"
L="$HERE/results/logs/hero8"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/launch.txt"; }
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
SMOKE="coding:1 tictactoe:1 eval_tampering:1"   # eval_tampering (20-step cap) is the plain-reply smoke: grade_your_own could take 4 h on gemma (9 Nov)

for m in $MODELS; do
  tag=qwen36; mc=0; [ "$m" = gemma-3-27b ] && { tag=gemma; mc=5; }
  if [ "$MODE" = smoke ]; then
    say "launching smoke for $m (stage hero8_smoke_$tag): $SMOKE"
    EXTRA_OV="$OV" MAXC=$mc GAP=${GAP:-12} bash "$HERE/scripts/modal_full.sh" "hero8_smoke_$tag" "$m" $SMOKE
    say "launched (hero8 smoke $m)"
    continue
  fi
  say "launching $m, hard rule + worked example (stage hero8_$tag): $MAIN"
  EXTRA_OV="$OV" MAXC=$mc GAP=${GAP:-12} bash "$HERE/scripts/modal_full.sh" "hero8_$tag" "$m" $MAIN
  say "launching ai_rd:10 for $m with the capped agent key (stage hero8_${tag}_key)"
  EXTRA_OV="$OV" MAXC=$mc GAP=${GAP:-12} MODAL_SECRETS=openrouter-agent bash "$HERE/scripts/modal_full.sh" "hero8_${tag}_key" "$m" ai_rd:10
  say "launching the other 20 for $m (stage hero8_${tag}_all): $OTHER + swe_rebench_v2:10"
  EXTRA_OV="$OV" MAXC=$mc GAP=${GAP:-12} bash "$HERE/scripts/modal_full.sh" "hero8_${tag}_all" "$m" $OTHER
  mkdir -p "$HERE/results/logs/hero8_${tag}_all"
  [ -f "$HERE/results/logs/hero8_${tag}_all/swe_rebench_v2__$m.log" ] && say "swe_rebench_v2 already launched for $m (log exists): skipping" || \
  ( cd "$HERE/agent-interp-envs" && nohup env -u VIRTUAL_ENV uv run modal run --detach ../scripts/swe_modal.py \
      --config "../configs/swe_rebench_v2__$m.yaml" --count 10 --stage "hero8_${tag}_all" --overrides "$OV" \
      > "$HERE/results/logs/hero8_${tag}_all/swe_rebench_v2__$m.log" 2>&1 & )
  say "launched (hero8 $m)"
done
say "all launched (hero8 $MODE $WHICH)"
