#!/usr/bin/env bash
# Fold in Qwen3.6 27B's solo runs (scripts/hero2_launch_qwen36.sh, 7 Nov 2026): waits for the launcher, then runs
# scripts/hero_solo_foldin.sh once per (model tag, stages), never two at once (it waits for the apps itself, fetches
# without --force, redacts, grades the judged environments within GRADE_CAP / CEIL, scores, audits). New file names only:
#   qwen3.6-27b        hero2_qwen36 + hero2_qwen36_key -> results/hero2_full_runs__qwen3.6-27b*.csv, hero2_loss_audit__qwen3.6-27b.md
#   qwen3.6-27b        hero2_qwen36_all                -> results/hero2_full_runs__qwen3.6-27b__other21*.csv, ..._audit__qwen3.6-27b__other21.md
#   qwen3.6-27b-think  hero2_qwen36_think              -> results/hero2_full_runs__qwen3.6-27b-think*.csv, hero2_loss_audit__qwen3.6-27b-think.md
# Absolute paths everywhere. Writes a DONE line to results/logs/hero2/foldin_qwen36.txt at the end.
#   CEIL=<$ since 18 Oct> nohup caffeinate -i bash /abs/.../scripts/hero2_foldin_qwen36.sh > /abs/log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
: "${CEIL:?set CEIL}"
GRADE_CAP=${GRADE_CAP:-100}; PASSES=${PASSES:-3}
L="$HERE/results/logs/hero2"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/foldin_qwen36.txt"; }
say "start (ceiling $CEIL, grade cap $GRADE_CAP)"
until grep -q "all four stages launched" "$L/launch_qwen36.txt" 2>/dev/null; do sleep 60; done
say "launcher done"
run_one () {   # model, stages, log-dir name, out prefix, audit name
  local m=$1 st=$2 LD="$HERE/results/logs/hero2/$3" pre=$4 aud=$5; mkdir -p "$LD"
  for pass in $(seq 1 "$PASSES"); do   # a later pass picks up what a pass could not fetch or grade (idempotent)
    while pgrep -f "scripts/hero_solo_foldin.sh" > /dev/null; do sleep 60; done
    say "$m ($st): fold-in pass $pass"
    WAIT_MINUTES=720 CEIL="$CEIL" GRADE_CAP="$GRADE_CAP" HERO_MODEL="$m" HERO_STAGES="$st" \
      HERO_OUT_PREFIX="$pre" HERO_AUDIT="$aud" HERO_LOG_DIR="$LD" \
      bash "$HERE/scripts/hero_solo_foldin.sh" > "$LD/foldin_pass$pass.out" 2>&1
    say "$m ($st): pass $pass exit $?; $(grep -h 'scored:\|STEP FAILED\|REACHED\|CROSSED' "$LD/progress.txt" 2>/dev/null | tail -3 | tr '\n' ';')"
    grep -q "STEP FAILED" <(tail -20 "$LD/progress.txt") || break
  done
}
run_one qwen3.6-27b-think hero2_qwen36_think qwen3.6-27b-think hero2_full_runs__qwen3.6-27b-think hero2_loss_audit__qwen3.6-27b-think.md
run_one qwen3.6-27b hero2_qwen36,hero2_qwen36_key qwen3.6-27b hero2_full_runs__qwen3.6-27b hero2_loss_audit__qwen3.6-27b.md
run_one qwen3.6-27b hero2_qwen36_all qwen3.6-27b__other21 hero2_full_runs__qwen3.6-27b__other21 hero2_loss_audit__qwen3.6-27b__other21.md
say "DONE"
