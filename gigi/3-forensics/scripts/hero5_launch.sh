#!/usr/bin/env bash
# hero5 (8 Nov 2026, GG: "come up with 5 different hard-hacks to test on a set of environments"): Llama 3.3 70B (fp16,
# CoreWeave) under the hard end rule plus one hack at a time (agent.hack, agent-interp-envs/src/agent_interp_envs/hacks.py,
# in patches/end_task_required.patch): do_the_work, no_early_exit, proof_of_work, plan_first, worked_example. Five tasks
# where "did it open the code" is measurable. Stages hero5_<hack>; the smoke test (one coding run per hack) goes to
# hero5_smoke_<hack>. Approved 8 Nov: estimate ~$400 incl. grading, cap $1,200 (watchdog: scripts/spend_watchdog.sh hero5).
#   nohup caffeinate -i bash /abs/.../scripts/hero5_launch.sh smoke|full > /abs/log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"; ROOT="$(cd ../.. && pwd)"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
MODE=${1:?smoke|full}; HACKS=${HACKS:-"do_the_work no_early_exit proof_of_work plan_first worked_example"}
L="$HERE/results/logs/hero5"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/launch_$MODE.txt"; }
case "$MODE" in
  smoke*) ENVS="coding:1"; PFX=hero5_${MODE}_ ;;   # smoke, smoke2, ...: one coding run per hack
  full)  ENVS="coding:10 coding_norules:10 dashboard_costly_failure:10 grade_your_own:5 optimize_rendering_engine:5"; PFX=hero5_ ;;
  *) echo "usage: $0 smoke|full" >&2; exit 2 ;;
esac
bal=$(FORENSICS_ENV_FILE="$ROOT/.env" python3 "$HERE/scripts/hero_usage.py" 2>/dev/null | grep -i "org credits" | grep -oE '\$[0-9][0-9,]*\.[0-9]+ remaining' | grep -oE '[0-9][0-9,]*\.[0-9]+' | tr -d ,)
say "org credit balance: ${bal:-unknown}"
python3 -c "import sys; sys.exit(0 if float('${bal:-0}') >= float('${MIN_CREDITS_USD:-300}') else 1)" || { say "balance below ${MIN_CREDITS_USD:-300} or unknown: NOT launching"; exit 1; }
for h in $HACKS; do
  say "launching ${PFX}${h}: $ENVS"
  EXTRA_OV="agent.end_task_required=true,agent.end_task_reprompts=10,agent.hack=$h" MAXC=0 GAP=${GAP:-12} \
    bash "$HERE/scripts/modal_full.sh" "${PFX}${h}" llama-3.3-70b $ENVS
done
say "launched ($MODE)"
