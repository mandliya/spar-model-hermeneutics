#!/usr/bin/env bash
# Fold in hero8 (gemma-3-27b and qwen3.6-27b, hard rule + worked example, all 29 tasks; launched by scripts/hero8_launch.sh
# full, 9 Nov 2026). For each model: waits for its "launched (hero8 <model>)" line, then runs scripts/hero_solo_foldin.sh
# (waits for the apps itself, fetches without --force, redacts, grades the judged environments within GRADE_CAP / CEIL,
# scores, audits) for the main stages and then for the other-21 stage, up to PASSES times each. Files:
# results/hero8_full_runs__<model>(.csv,_per_run.csv) + results/hero8_loss_audit__<model>.md for the 8 main tasks; the
# same with __other21 for the rest; logs in results/logs/hero8/<model>(__other21)/. Writes DONE to
# results/logs/hero8/foldin.txt. Does not commit.
#   CEIL=<$ since 18 Oct ceiling> MODELS="gemma-3-27b qwen3.6-27b" nohup caffeinate -i bash /abs/.../scripts/hero8_foldin.sh > /abs/log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
: "${CEIL:?set CEIL}"; GRADE_CAP=${GRADE_CAP:-60}; PASSES=${PASSES:-3}; MODELS=${MODELS:-"gemma-3-27b qwen3.6-27b"}
L="$HERE/results/logs/hero8"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/foldin.txt"; }
say "start (ceiling $CEIL, grade cap $GRADE_CAP, models: $MODELS)"
run_stage () {  # $1 model, $2 stages, $3 out prefix, $4 audit, $5 log dir, $6 extra env (HERO_ALL_ENVS=1 or empty)
  local LD="$L/$5"; mkdir -p "$LD"
  for pass in $(seq 1 "$PASSES"); do
    while pgrep -f "scripts/hero_solo_foldin.sh" > /dev/null; do sleep 60; done
    say "$1 $2: fold-in pass $pass"
    env ${6:-} WAIT_MINUTES=600 CEIL="$CEIL" GRADE_CAP="$GRADE_CAP" HERO_MODEL="$1" HERO_STAGES="$2" \
      HERO_OUT_PREFIX="$3" HERO_AUDIT="$4" HERO_LOG_DIR="$LD" \
      bash "$HERE/scripts/hero_solo_foldin.sh" > "$LD/foldin_pass$pass.out" 2>&1
    say "$1 $2: pass $pass exit $?; $(grep -h 'scored:\|STEP FAILED\|REACHED\|CROSSED' "$LD/progress.txt" 2>/dev/null | tail -3 | tr '\n' ';')"
    grep -q "STEP FAILED" <(tail -20 "$LD/progress.txt" 2>/dev/null) || break
  done
}
for m in $MODELS; do
  tag=qwen36; [ "$m" = gemma-3-27b ] && tag=gemma
  until grep -q "launched (hero8 $m)" "$L/launch.txt" 2>/dev/null; do
    grep -q "NOT launching" "$L/launch.txt" 2>/dev/null && { say "launcher refused: nothing to fold in for $m"; continue 2; }
    sleep 60
  done
  say "launcher done for $m"
  run_stage "$m" "hero8_${tag},hero8_${tag}_key" "hero8_full_runs__$m" "hero8_loss_audit__$m.md" "$m" ""
  run_stage "$m" "hero8_${tag}_all" "hero8_full_runs__${m}__other21" "hero8_loss_audit__${m}__other21.md" "${m}__other21" "HERO_ALL_ENVS=1"
done
say "DONE"
