#!/usr/bin/env bash
# move_long.sh (17 Nov 2026, run at 15:00 UTC from /workspace/move_long.sh). Two r2 apps run long tasks one or two at a time and would finish many hours
# after everything else: grade_your_own (~1 h a run, 2 at a time, ~34 runs to go = ~17 h) and
# optimize_rendering_engine_graded (~0.6 h a run, 1 at a time, ~14 to go = ~8 h). Their launchers on the box stop, so
# they start no new runs; the runs already in flight finish on their own (the Modal apps are detached, nothing is cut).
# The runs they had not started go to the head of the paced queue (classic_fleet3.sh) as chunks of at most 3.
set -uo pipefail
H=/workspace/forensics/gigi/3-forensics; S=/workspace/logs/stages.txt; Q=/workspace/logs/classic_queue.txt
say () { echo "$(date -u +%FT%TZ) $*" >> "$S"; }
quiet () {   # the supervisor has just finished a cycle: last line "in flight: ...", under 90 s old (it sleeps 300 s)
  for i in $(seq 1 100); do
    last=$(tail -n 1 "$S")
    case "$last" in *" in flight: "*)
      age=$(( $(date +%s) - $(date -d "${last%% *}" +%s) )); [ "$age" -lt 90 ] && return 0;; esac
    sleep 10
  done; return 1
}
chunks () {   # chunks <stage prefix> <first k> <task> <runs> <max per chunk>
  local p=$1 k=$2 e=$3 n=$4 m=$5 c
  while [ "$n" -gt 0 ]; do c=$(( (n + (n + m - 1) / m - 1) / ((n + m - 1) / m) )); echo "${p}_$k $e $c"; n=$((n - c)); k=$((k + 1)); done
}
cd "$H/agent-interp-envs"; export MODAL_PROFILE=spar
quiet || { say "move_long: no quiet moment found, nothing changed"; exit 1; }
for e in grade_your_own optimize_rendering_engine_graded; do
  pids=$(pgrep -f "results/raw/classic_qwen36_r2/${e}\$" | tr '\n' ' ')
  echo "$e launcher pids: $pids ($(for p in $pids; do cat /proc/$p/comm 2>/dev/null; done | tr '\n' ' '))"
  [ -n "$pids" ] && kill $pids
done
sleep 20
for e in grade_your_own optimize_rendering_engine_graded; do
  left=$(pgrep -f "results/raw/classic_qwen36_r2/${e}\$" | tr '\n' ' '); [ -n "$left" ] && { echo "still alive: $e $left, SIGKILL"; kill -9 $left; }
done
sleep 60   # let any launch that was in progress land on the volume before counting
out=$(timeout 300 env -u VIRTUAL_ENV uv run python ../scripts/classic_launched.py classic_qwen36_r2/grade_your_own classic_qwen36_r2/optimize_rendering_engine_graded 2>&1)
echo "$out"
g=$(echo "$out" | awk '/grade_your_own launched/ {print $3}'); o=$(echo "$out" | awk '/optimize_rendering_engine_graded launched/ {print $3}')
case "$g$o" in ''|*[!0-9]*) say "move_long: COULD NOT COUNT the launched runs ($out); launchers stopped, queue NOT changed: fix by hand"; exit 1;; esac
ng=$((46 - g)); no=$((23 - o))
quiet || { say "move_long: launchers stopped but no quiet moment to edit the queue: queue NOT changed, fix by hand"; exit 1; }
{ chunks classic_qwen36_r3 1 grade_your_own $ng 3; chunks classic_qwen36_r3 5 optimize_rendering_engine_graded $no 3; cat "$Q"; } > "$Q.new" && mv "$Q.new" "$Q"
say "moved: classic_qwen36_r2 grade_your_own and optimize_rendering_engine_graded start no new runs (launched $g of 46 and $o of 23; those in flight finish on their own). The other $ng and $no runs are at the head of the queue in chunks of at most 3 ($(grep -c . "$Q") chunks queued now)"
head -n 20 "$Q"
