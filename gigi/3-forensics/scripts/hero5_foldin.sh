#!/usr/bin/env bash
# Fold in the five-hack test (hero5, 8 Nov 2026): waits for scripts/hero5_launch.sh full to finish launching, then for each
# hack runs scripts/hero_solo_foldin.sh (waits for the apps, fetches without --force, redacts, grades the judged tasks within
# GRADE_CAP / CEIL, scores, audits) up to PASSES times, then renders scripts/hero5_charts.py. Files per hack:
# results/hero5_full_runs__llama-3.3-70b__<hack>(.csv,_per_run.csv), results/hero5_loss_audit__llama-3.3-70b__<hack>.md,
# logs results/logs/hero5/llama-3.3-70b__<hack>/. Writes DONE to results/logs/hero5/foldin.txt. Does not commit.
#   CEIL=<$ since 18 Oct ceiling> nohup caffeinate -i bash /abs/.../scripts/hero5_foldin.sh > /abs/log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
: "${CEIL:?set CEIL}"; GRADE_CAP=${GRADE_CAP:-60}; PASSES=${PASSES:-3}
HACKS=${HACKS:-"do_the_work no_early_exit proof_of_work plan_first worked_example"}
L="$HERE/results/logs/hero5"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/foldin.txt"; }
say "start (ceiling $CEIL, grade cap $GRADE_CAP)"
until grep -q "launched (full)" "$L/launch_full.txt" 2>/dev/null; do
  grep -q "NOT launching" "$L/launch_full.txt" 2>/dev/null && { say "launcher refused: nothing to fold in"; exit 1; }
  sleep 60
done
say "launcher done"
for h in $HACKS; do
  LD="$L/llama-3.3-70b__$h"; mkdir -p "$LD"
  for pass in $(seq 1 "$PASSES"); do
    while pgrep -f "scripts/hero_solo_foldin.sh" > /dev/null; do sleep 60; done
    say "$h: fold-in pass $pass"
    WAIT_MINUTES=600 CEIL="$CEIL" GRADE_CAP="$GRADE_CAP" HERO_MODEL=llama-3.3-70b HERO_STAGES="hero5_$h" \
      HERO_OUT_PREFIX="hero5_full_runs__llama-3.3-70b__$h" HERO_AUDIT="hero5_loss_audit__llama-3.3-70b__$h.md" HERO_LOG_DIR="$LD" \
      bash "$HERE/scripts/hero_solo_foldin.sh" > "$LD/foldin_pass$pass.out" 2>&1
    say "$h: pass $pass exit $?; $(grep -h 'scored:\|STEP FAILED\|REACHED\|CROSSED' "$LD/progress.txt" 2>/dev/null | tail -3 | tr '\n' ';')"
    grep -q "STEP FAILED" <(tail -20 "$LD/progress.txt" 2>/dev/null) || break
  done
done
( cd "$HERE/agent-interp-envs" && env -u VIRTUAL_ENV uv run --with matplotlib python ../scripts/hero5_charts.py > "$L/charts.out" 2>&1 ) && say "charts rendered" || say "charts FAILED (see charts.out)"
say "DONE"
