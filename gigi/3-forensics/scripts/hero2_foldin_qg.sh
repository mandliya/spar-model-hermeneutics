#!/usr/bin/env bash
# Fold in Qwen3-Next-80B's and gemma-3-27b's other-21-environment runs with the end-of-run check (stages hero2_qwen_all,
# hero2_gemma_all; launched by scripts/hero2_launch_other_envs_qg.sh, 6 Nov 2026). Waits for the launcher, then runs
# scripts/hero_solo_foldin.sh once per model, never two at once (it waits for the apps itself, fetches without --force,
# redacts, grades the judged environments with Opus 5 then Opus 4.8 within GRADE_CAP / CEIL, scores, audits). New file
# names only. Absolute paths everywhere. Writes a DONE line to results/logs/hero2/foldin_qg.txt at the end.
#   CEIL=<$ since 18 Oct> nohup caffeinate -i bash /abs/.../scripts/hero2_foldin_qg.sh > /abs/log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
: "${CEIL:?set CEIL}"
GRADE_CAP=${GRADE_CAP:-100}; PASSES=${PASSES:-3}
L="$HERE/results/logs/hero2"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/foldin_qg.txt"; }
say "start (ceiling $CEIL, grade cap $GRADE_CAP)"
until grep -q "both models launched" "$L/launch_other_envs_qg.txt" 2>/dev/null; do sleep 60; done
say "launcher done"
for m in qwen3-next-80b gemma-3-27b; do
  tag=qwen; [ "$m" = gemma-3-27b ] && tag=gemma
  LD="$HERE/results/logs/hero2/${m}__other21"; mkdir -p "$LD"
  for pass in $(seq 1 "$PASSES"); do   # a second pass picks up what a pass could not fetch or grade (idempotent)
    while pgrep -f "scripts/hero_solo_foldin.sh" > /dev/null; do sleep 60; done
    say "$m: fold-in pass $pass"
    WAIT_MINUTES=600 CEIL="$CEIL" GRADE_CAP="$GRADE_CAP" HERO_MODEL="$m" HERO_STAGES="hero2_${tag}_all" \
      HERO_OUT_PREFIX="hero2_full_runs__${m}__other21" HERO_AUDIT="hero2_loss_audit__${m}__other21.md" HERO_LOG_DIR="$LD" \
      bash "$HERE/scripts/hero_solo_foldin.sh" > "$LD/foldin_pass$pass.out" 2>&1
    say "$m: pass $pass exit $?; $(grep -h 'scored:\|STEP FAILED\|REACHED\|CROSSED' "$LD/progress.txt" 2>/dev/null | tail -3 | tr '\n' ';')"
    grep -q "STEP FAILED" <(tail -20 "$LD/progress.txt") || break
  done
done
say "DONE"
