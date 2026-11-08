#!/usr/bin/env bash
# Fold in hero7 (Llama 3.3 70B, hard rule + CONTROL worked example, all 29 tasks; launched by scripts/hero7_launch.sh,
# 8 Nov 2026 night). Waits for the launcher, then runs scripts/hero_solo_foldin.sh (waits for the apps itself, fetches
# without --force, redacts, grades the judged environments within GRADE_CAP / CEIL, scores, audits) for the main stages
# and then for the other-21 stage, up to PASSES times each. Files: results/hero7_full_runs__llama-3.3-70b(.csv,_per_run.csv)
# + results/hero7_loss_audit__llama-3.3-70b.md for the 8 main tasks; the same with __other21 for the rest; logs in
# results/logs/hero7/llama-3.3-70b(__other21)/. Writes DONE to results/logs/hero7/foldin.txt. Does not commit.
#   CEIL=<$ since 18 Oct ceiling> nohup caffeinate -i bash /abs/.../scripts/hero7_foldin.sh > /abs/log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
: "${CEIL:?set CEIL}"; GRADE_CAP=${GRADE_CAP:-60}; PASSES=${PASSES:-3}
L="$HERE/results/logs/hero7"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/foldin.txt"; }
say "start (ceiling $CEIL, grade cap $GRADE_CAP)"
until grep -q "launched (hero7)" "$L/launch.txt" 2>/dev/null; do
  grep -q "NOT launching" "$L/launch.txt" 2>/dev/null && { say "launcher refused: nothing to fold in"; exit 1; }
  sleep 60
done
say "launcher done"
run_stage () {  # $1 stages, $2 out prefix, $3 audit, $4 log dir, $5 extra env (HERO_ALL_ENVS=1 or empty)
  local LD="$L/$4"; mkdir -p "$LD"
  for pass in $(seq 1 "$PASSES"); do
    while pgrep -f "scripts/hero_solo_foldin.sh" > /dev/null; do sleep 60; done
    say "$1: fold-in pass $pass"
    env ${5:-} WAIT_MINUTES=600 CEIL="$CEIL" GRADE_CAP="$GRADE_CAP" HERO_MODEL=llama-3.3-70b HERO_STAGES="$1" \
      HERO_OUT_PREFIX="$2" HERO_AUDIT="$3" HERO_LOG_DIR="$LD" \
      bash "$HERE/scripts/hero_solo_foldin.sh" > "$LD/foldin_pass$pass.out" 2>&1
    say "$1: pass $pass exit $?; $(grep -h 'scored:\|STEP FAILED\|REACHED\|CROSSED' "$LD/progress.txt" 2>/dev/null | tail -3 | tr '\n' ';')"
    grep -q "STEP FAILED" <(tail -20 "$LD/progress.txt" 2>/dev/null) || break
  done
}
run_stage "hero7_control,hero7_control_key" "hero7_full_runs__llama-3.3-70b" "hero7_loss_audit__llama-3.3-70b.md" "llama-3.3-70b" ""
run_stage "hero7_control_all" "hero7_full_runs__llama-3.3-70b__other21" "hero7_loss_audit__llama-3.3-70b__other21.md" "llama-3.3-70b__other21" "HERO_ALL_ENVS=1"
say "DONE"
