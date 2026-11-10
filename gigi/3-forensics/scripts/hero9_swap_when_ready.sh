#!/usr/bin/env bash
# Launch the symmetric hero swap by itself, but only once BOTH gates are open (10 Nov 2026):
#   1. the hero9 no-check fleets are folded in: a line "DONE" in results/logs/hero9/until_done.txt or results/logs/hero9/foldin.txt;
#   2. GG approved the estimate: the file results/logs/hero9/SWAP_APPROVED exists. The parent session creates it after GG's
#      yes; its first number, if any, is the approved estimate in dollars (the watchdog cap is 3 x that; CAP_USD overrides).
# Then: the plan (scripts/hero9_swap_plan.py, exclusive create; an existing plan is used as is), a free dry run of the
# launcher (merged configs), and the launch (scripts/hero9_swap.sh --cap). Detached and sleep-safe (nohup caffeinate -i),
# polls every 120 s, idempotent (a launched swap is left alone). Spends nothing until SWAP_APPROVED exists.
# Log: results/logs/hero9_swap/when_ready.txt.
#   bash gigi/3-forensics/scripts/hero9_swap_when_ready.sh      detaches itself, returns at once
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
L="$HERE/results/logs/hero9_swap"; mkdir -p "$L"; G="$HERE/results/logs/hero9"
say () { echo "$(date '+%F %T') $*" >> "$L/when_ready.txt"; }
if [ -z "${HERO9_WHEN_READY_CHILD:-}" ]; then
  if [ -f "$L/when_ready.pid" ] && kill -0 "$(cat "$L/when_ready.pid")" 2>/dev/null; then echo "hero9_swap_when_ready: already running (pid $(cat "$L/when_ready.pid"))"; exit 0; fi
  HERO9_WHEN_READY_CHILD=1 nohup caffeinate -i bash "$0" "$@" >> "$L/when_ready_stdout.txt" 2>&1 &
  disown 2>/dev/null; echo "hero9_swap_when_ready: detached (pid $!); log $L/when_ready.txt"; exit 0
fi
echo $$ > "$L/when_ready.pid"; trap 'rm -f "$L/when_ready.pid"' EXIT
say "start: waiting for DONE in $G/until_done.txt or $G/foldin.txt AND the file $G/SWAP_APPROVED"
i=0
until { grep -q "DONE" "$G/until_done.txt" 2>/dev/null || grep -q "DONE" "$G/foldin.txt" 2>/dev/null; } && [ -f "$G/SWAP_APPROVED" ]; do
  i=$((i + 1)); [ $((i % 30)) = 1 ] && say "waiting (fleets done: $({ grep -q DONE "$G/until_done.txt" 2>/dev/null || grep -q DONE "$G/foldin.txt" 2>/dev/null; } && echo yes || echo no); approved: $([ -f "$G/SWAP_APPROVED" ] && echo yes || echo no))"
  sleep 120
done
say "both gates open: $(head -c 200 "$G/SWAP_APPROVED" | tr '\n' ' ')"
if grep -q "all launched" "$L/launch.txt" 2>/dev/null; then say "the swap is launched already: nothing to do"; exit 0; fi
PLAN="$HERE/results/hero9_swap_plan.json"
if [ -f "$PLAN" ]; then say "plan exists, used as is: $PLAN"; else
  (cd "$HERE/agent-interp-envs" && env -u VIRTUAL_ENV uv run python ../scripts/hero9_swap_plan.py > "$L/plan_build.txt" 2>&1) && [ -f "$PLAN" ] \
    || { say "PLAN NOT WRITTEN: $(tail -3 "$L/plan_build.txt" | tr '\n' ' '); nothing launched"; exit 1; }
  say "plan written: $(head -1 "$L/plan_build.txt"); $(grep '^estimate' "$L/plan_build.txt" | head -1)"
fi
APPROVED=$(grep -oE '[0-9][0-9,]*(\.[0-9]+)?' "$G/SWAP_APPROVED" | head -1 | tr -d ,)
EST=$(python3 -c "import json; print(json.load(open('$HERE/results/hero9_swap_estimate.json'))['total_usd'])" 2>/dev/null || echo "")
CAP=${CAP_USD:-$(python3 -c "print(round(3 * float('${APPROVED:-${EST:-0}}')))")}
[ "${CAP:-0}" -gt 0 ] 2>/dev/null || { say "no cap: SWAP_APPROVED has no number and no estimate file; set CAP_USD; nothing launched"; exit 1; }
say "cap \$$CAP (approved ${APPROVED:-none}, estimate ${EST:-?}, x3)"
bash "$HERE/scripts/hero9_swap.sh" --dry-run > "$L/launch_dry.txt" 2>&1 || { say "launcher dry run FAILED: $(tail -3 "$L/launch_dry.txt" | tr '\n' ' '); nothing launched"; exit 1; }
say "launcher dry run OK: $(head -1 "$L/launch_dry.txt")"
bash "$HERE/scripts/hero9_swap.sh" --cap "$CAP" > "$L/launch_real.txt" 2>&1; rc=$?
say "hero9_swap.sh exit $rc: $(head -1 "$L/launch_real.txt"); launch log $L/launch.txt; fold-in log $L/foldin.txt"
exit $rc
