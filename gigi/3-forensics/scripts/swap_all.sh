#!/usr/bin/env bash
# The swap in every environment (plan: results/swap_all_plan.json from scripts/swap_all_plan.py): each restart
# point resumed on Modal twice, with Qwen (the reference, its own config) and with gemma swapped in (text mode),
# from the current tree (--use-current), every continuation started at once. 4 launches in parallel, each
# retried while Modal's app-creation limit turns it away. Results: results/raw/swap_all/<env>/<point>/<who>/;
# logs: results/logs/swap_all/.   MODAL_PROFILE=... MODAL_SECRET_SUFFIX=... MODAL_VOLUME_PREFIX=... bash swap_all.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; LOG="$HERE/results/logs/swap_all"; mkdir -p "$LOG"
GEMMA="agent.model=google/gemma-3-27b-it,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null,agent.text_tools=true"
CMDS=${1:-$LOG/commands.txt}   # optional: a ready-made command file (e.g. commands_fit.txt), else built from the plan
[ -n "${1:-}" ] || python3 - "$HERE" "$GEMMA" > "$LOG/commands.txt" <<'PY'
import json, sys, pathlib, shlex
here, gemma = pathlib.Path(sys.argv[1]), sys.argv[2]
for p in json.load(open(here / "results/swap_all_plan.json")):
    run = here / p["run"]; point = f"{run.parent.name}_{run.name}_K{p['K']}"
    for who, ov in (("qwen", ""), ("gemma", gemma)):
        out = here / "results/raw/swap_all" / p["env"] / point / who
        log = here / "results/logs/swap_all" / f"{p['env']}__{point}__{who}.log"
        args = ["--step-dir", str(run / f"step-{p['K']}"), "--count", str(p["reps"]), "--use-current", "--max-concurrent", "0",
                "--no-judge", "--timeout", "7200", "--results-dir", str(out)] + (["--overrides", ov] if ov else [])
        print(shlex.quote(str(log)) + " " + " ".join(shlex.quote(a) for a in args))
PY
echo "$(date '+%F %T') launching $(wc -l < "$CMDS") resumes from $(basename "$CMDS")" >> "$LOG/launch.txt"
launch_one () {
  log=$1; shift
  [ -f "$log" ] && grep -q "Launching\|Resuming\|Checkpoint:" "$log" && ! grep -q "rate limit exceeded" "$log" && return 0
  for try in 1 2 3 4 5 6 7 8; do
    ( cd "$HERE/agent-interp-envs" && MODAL_SECRETS=openrouter-secret nohup env -u VIRTUAL_ENV uv run modal run --detach \
        scripts/resume_modal.py "$@" > "$log" 2>&1 & ); sleep 25
    grep -q "rate limit exceeded" "$log" || break
    sleep $((10 * try))
  done
  echo "$(date '+%F %T') launched $(basename "$log")" >> "$(dirname "$log")/launch.txt"
}
export -f launch_one; export HERE
xargs -P 4 -L 1 bash -c 'eval "launch_one $*"' _ < "$CMDS"
echo "$(date '+%F %T') all launched ($(basename "$CMDS"))" >> "$LOG/launch.txt"
