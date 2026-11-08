#!/usr/bin/env bash
# Fold in the hero4_llama_fix rerun (scripts/hero4_launch_llama_fix.sh, 8 Nov 2026). Waits for the launcher, then runs
# scripts/hero_solo_foldin.sh (waits for the apps, fetches without --force, redacts, grades swe within GRADE_CAP / CEIL,
# scores all 29 environments because HERO_ALL_ENVS=1, audits), up to PASSES times. Files:
# results/hero4_full_runs__llama-3.3-70b__fix*.csv, results/hero4_loss_audit__llama-3.3-70b__fix.md, logs in
# results/logs/hero4/llama-3.3-70b__fix/. Writes DONE to results/logs/hero4/foldin_fix.txt. Does not commit.
#   CEIL=<$ since 18 Oct + margin> nohup caffeinate -i bash /abs/.../scripts/hero4_foldin_llama_fix.sh > /abs/log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
: "${CEIL:?set CEIL}"
GRADE_CAP=${GRADE_CAP:-30}; PASSES=${PASSES:-3}
L="$HERE/results/logs/hero4"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/foldin_fix.txt"; }
say "start (ceiling $CEIL, grade cap $GRADE_CAP)"
until grep -q "launched (hero4_llama_fix)" "$L/launch_fix.txt" 2>/dev/null; do
  grep -q "NOT launching" "$L/launch_fix.txt" 2>/dev/null && { say "launcher refused: nothing to fold in"; exit 1; }
  sleep 60
done
say "launcher done"
LD="$L/llama-3.3-70b__fix"; mkdir -p "$LD"
for pass in $(seq 1 "$PASSES"); do
  while pgrep -f "scripts/hero_solo_foldin.sh" > /dev/null; do sleep 60; done
  say "fold-in pass $pass"
  WAIT_MINUTES=600 CEIL="$CEIL" GRADE_CAP="$GRADE_CAP" HERO_ALL_ENVS=1 HERO_MODEL=llama-3.3-70b HERO_STAGES=hero4_llama_fix \
    HERO_OUT_PREFIX="hero4_full_runs__llama-3.3-70b__fix" HERO_AUDIT="hero4_loss_audit__llama-3.3-70b__fix.md" HERO_LOG_DIR="$LD" \
    bash "$HERE/scripts/hero_solo_foldin.sh" > "$LD/foldin_pass$pass.out" 2>&1
  say "pass $pass exit $?; $(grep -h 'scored:\|STEP FAILED\|REACHED\|CROSSED' "$LD/progress.txt" 2>/dev/null | tail -3 | tr '\n' ';')"
  grep -q "STEP FAILED" <(tail -20 "$LD/progress.txt") || break
done
say "DONE"
