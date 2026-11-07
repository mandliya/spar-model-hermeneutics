#!/usr/bin/env bash
# Fold in Llama's other-21 runs with the hard end rule (stage hero4_llama_all, launched by scripts/hero4_launch_llama_all.sh,
# 7 Nov 2026). Waits for the launcher, then runs scripts/hero_solo_foldin.sh (it waits for the apps itself, fetches without
# --force, redacts, grades the judged environments within GRADE_CAP / CEIL, scores, audits), up to PASSES times so a pass
# that could not fetch or grade something is retried. New file names only (results/hero4_full_runs__llama-3.3-70b__other21*.csv,
# results/hero4_loss_audit__llama-3.3-70b__other21.md, logs in results/logs/hero4/llama-3.3-70b__other21/). Absolute paths.
# Writes a DONE line to results/logs/hero4/foldin_all.txt at the end. Does not commit (the parent session does).
#   CEIL=<$ since 18 Oct> nohup caffeinate -i bash /abs/.../scripts/hero4_foldin_llama_all.sh > /abs/log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
: "${CEIL:?set CEIL}"
GRADE_CAP=${GRADE_CAP:-60}; PASSES=${PASSES:-3}
L="$HERE/results/logs/hero4"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/foldin_all.txt"; }
say "start (ceiling $CEIL, grade cap $GRADE_CAP)"
until grep -q "launched (hero4_llama_all)" "$L/launch_all.txt" 2>/dev/null; do
  grep -q "NOT launching" "$L/launch_all.txt" 2>/dev/null && { say "launcher refused: nothing to fold in"; exit 1; }
  sleep 60
done
say "launcher done"
LD="$L/llama-3.3-70b__other21"; mkdir -p "$LD"
for pass in $(seq 1 "$PASSES"); do
  while pgrep -f "scripts/hero_solo_foldin.sh" > /dev/null; do sleep 60; done
  say "fold-in pass $pass"
  WAIT_MINUTES=600 CEIL="$CEIL" GRADE_CAP="$GRADE_CAP" HERO_MODEL=llama-3.3-70b HERO_STAGES=hero4_llama_all \
    HERO_OUT_PREFIX="hero4_full_runs__llama-3.3-70b__other21" HERO_AUDIT="hero4_loss_audit__llama-3.3-70b__other21.md" HERO_LOG_DIR="$LD" \
    bash "$HERE/scripts/hero_solo_foldin.sh" > "$LD/foldin_pass$pass.out" 2>&1
  say "pass $pass exit $?; $(grep -h 'scored:\|STEP FAILED\|REACHED\|CROSSED' "$LD/progress.txt" 2>/dev/null | tail -3 | tr '\n' ';')"
  grep -q "STEP FAILED" <(tail -20 "$LD/progress.txt") || break
done
say "DONE"
