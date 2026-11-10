#!/usr/bin/env bash
# The symmetric hero swap (10 Nov 2026 call; plan: results/hero9_swap_plan.json from scripts/hero9_swap_plan.py).
# Modelled on scripts/hero_swap.sh: every restart point is resumed on Modal once per arm from the current tree
# (--use-current), every continuation of a point started at once (--max-concurrent 0), 4 launches in parallel, each
# retried while Modal's app-creation limit turns it away. Each point names its source hero (`src`) and two arms:
#   the source itself   no override: the checkpoint fleet's own config.yaml (the plain no-check run's config)
#   the other hero      the override string below (plain: no end check, no hard rule, no hack)
#     qwen36  Qwen3.6 27B, native tools, thinking off, DeepInfra fp8 (no full-precision endpoint exists)
#     gemma   gemma-3-27b-it, text-mode tools, DeepInfra fp8
# Results: results/raw/hero9_swap/<env>/<point>/<who>/; logs: results/logs/hero9_swap/ (arms.txt records each arm's
# override string and precision at every launch). <point> = <src>_<fleet stamp>_run-N_K<K>, so the two directions never
# collide. A spend watchdog (scripts/spend_watchdog.sh hero9_swap, CAP_USD = --cap) stops our resume apps once the key
# has spent the cap; the fold-in (scripts/hero9_swap_foldin.sh) is started detached after the launch.
#   bash hero9_swap.sh --dry-run [--plan <file>]      write the command file to a temp file, check both directions'
#                                                     merged configs (free, no network), launch nothing
#   bash hero9_swap.sh --cap <USD> [--plan <file>]    launch (every launch spends money: GG approves the estimate first,
#                                                     scripts/hero9_swap_when_ready.sh gates this on results/logs/hero9/SWAP_APPROVED)
# Env: SWAP_TIMEOUT (seconds per continuation, default 14400: Qwen3.6 works 100+ steps in the long tasks), CEIL (the
# fold-in's grading ceiling, $ of the key since 18 Oct; default: CAP + the key's spend now), GRADE_CAP (default 60).
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
SWAPNAME=hero9_swap; LOG="$HERE/results/logs/$SWAPNAME"; RAW="$HERE/results/raw/$SWAPNAME"
export MODAL_PROFILE=${MODAL_PROFILE:-spar} MODAL_SECRET_SUFFIX=${MODAL_SECRET_SUFFIX:--gigi} MODAL_VOLUME_PREFIX=${MODAL_VOLUME_PREFIX:-gigi-forensics-}

# ---- override strings (hero_swap.sh's, plus agent.end_confirm=false and agent.end_task_required=false: the other hero
# always runs plain, whatever the checkpoint fleet's config carried) ----
QWEN36="agent.model=qwen/qwen3.6-27b,agent.temperature=0.7,agent.top_p=0.8,agent.top_k=20,agent.min_p=0,agent.text_tools=false,agent.reasoning_effort=disabled,agent.provider_preferences.only=[deepinfra],agent.provider_preferences.quantizations=[fp8],agent.provider_preferences.allow_fallbacks=false,agent.end_confirm=false,agent.end_task_required=false"
GEMMA="agent.model=google/gemma-3-27b-it,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null,agent.text_tools=true,agent.reasoning_effort=none,agent.provider_preferences.only=[deepinfra],agent.provider_preferences.quantizations=[fp8],agent.provider_preferences.allow_fallbacks=false,agent.provider_preferences.require_parameters=true,agent.end_confirm=false,agent.end_task_required=false"

PLAN="$HERE/results/${SWAPNAME}_plan.json"; DRY=0; CAP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --plan) PLAN="$2"; case "$PLAN" in /*) ;; *) PLAN="$HERE/$PLAN" ;; esac; shift ;;
    --dry-run) DRY=1 ;;
    --cap) CAP="$2"; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
[ -f "$PLAN" ] || { echo "no plan at $PLAN: run scripts/hero9_swap_plan.py first" >&2; exit 1; }
if [ $DRY = 0 ]; then
  [ -n "$CAP" ] || { echo "--cap <USD> is required for a real launch (the approved estimate x 3)" >&2; exit 2; }
  mkdir -p "$LOG"
  if grep -q "all launched" "$LOG/launch.txt" 2>/dev/null; then echo "hero9_swap: launched already ($LOG/launch.txt): nothing to do"; exit 0; fi
fi
STAMP=$(date '+%Y%m%d-%H%M%S')
if [ $DRY = 1 ]; then CMDS=$(mktemp "${TMPDIR:-/tmp}/hero9_swap_commands.XXXXXX"); else CMDS="$LOG/commands_$STAMP.txt"; fi
python3 - "$HERE" "$PLAN" "$QWEN36" "$GEMMA" "$SWAPNAME" > "$CMDS" <<'PY' || exit 1
import json, os, pathlib, shlex, sys
here, plan, swapname = pathlib.Path(sys.argv[1]), json.load(open(sys.argv[2])), sys.argv[5]
OV = {"qwen36": sys.argv[3], "gemma": sys.argv[4]}
for p in plan:
    run = here / p["run"]; point = f"{p['src']}_{run.parent.name}_{run.name}_K{p['K']}"
    for who in p["arms"]:
        ov = "" if who == p["src"] else OV[who]
        out = here / "results/raw" / swapname / p["env"] / point / who
        log = here / "results/logs" / swapname / f"{p['env']}__{point}__{who}.log"
        args = ["--step-dir", str(run / f"step-{p['K']}"), "--count", str(p.get("target", p["reps"])), "--use-current", "--max-concurrent", "0",
                "--no-judge", "--timeout", os.environ.get("SWAP_TIMEOUT", "14400"), "--results-dir", str(out)] + (["--overrides", ov] if ov else [])
        print(shlex.quote(str(log)) + " " + " ".join(shlex.quote(a) for a in args))
PY
N_RES=$(wc -l < "$CMDS" | tr -d ' ')
N_TRIES=$(grep -o -- "--count [0-9]*" "$CMDS" | awk '{s += $2} END {print s + 0}')
echo "$N_RES resumes, $N_TRIES continuations -> $CMDS"
for a in gemma qwen36; do
  printf "  %-7s %5s continuations (%s as the source itself, %s as the other hero)\n" "$a" \
    "$(grep -E -- "__${a}\.log " "$CMDS" | grep -o -- "--count [0-9]*" | awk '{s += $2} END {print s + 0}')" \
    "$(grep -E -- "__${a}_[^ ]*__${a}\.log " "$CMDS" | wc -l | tr -d ' ')" "$(grep -E -- "__${a}\.log " "$CMDS" | grep -vE -- "__${a}_[^ ]*__${a}\.log " | wc -l | tr -d ' ')"
done

if [ $DRY = 1 ]; then
  # free check: each other-hero override merged onto a real checkpoint fleet config of the opposite source, exactly as
  # resume_modal.py does (run_common.load_config); the self arm is the base as it is
  python3 - "$HERE" "$PLAN" <<'PY' > "$CMDS.bases"
import json, pathlib, sys
here, plan = pathlib.Path(sys.argv[1]), json.load(open(sys.argv[2]))
for src in ("gemma", "qwen36"):
    p = next((p for p in plan if p["src"] == src), None)
    print(src, (here / p["run"]).parent / "config.yaml" if p else "")
PY
  ( cd "$HERE/agent-interp-envs" && env -u VIRTUAL_ENV uv run python - "$CMDS.bases" "$QWEN36" "$GEMMA" <<'PY'
import sys
from omegaconf import OmegaConf
sys.path.insert(0, "scripts")
from run_common import load_config, parse_overrides
OV = {"qwen36": sys.argv[2], "gemma": sys.argv[3]}; other = {"gemma": "qwen36", "qwen36": "gemma"}
KEYS = ("model", "temperature", "top_p", "top_k", "min_p", "text_tools", "reasoning_effort", "provider_preferences", "end_confirm", "end_task_required", "hack")
for line in open(sys.argv[1]):
    src, _, cfg = line.strip().partition(" ")
    if not cfg:
        print(f"  {src}: no point in the plan, nothing to check"); continue
    base = OmegaConf.to_container(load_config(cfg, None).agent)
    print(f"  source {src}, base {cfg}")
    print(f"    self  {src:7s}", {k: base.get(k) for k in KEYS})
    for k in ("end_confirm", "end_task_required", "hack"):
        if base.get(k):
            print(f"    ! the base config has {k}={base.get(k)}: the self arm inherits it (fine for a code test on hero2 runs; a hero9 run must be plain)")
    ag = OmegaConf.to_container(load_config(cfg, parse_overrides(OV[other[src]])).agent)
    print(f"    other {other[src]:7s}", {k: ag.get(k) for k in KEYS})
    assert ag["provider_preferences"].get("allow_fallbacks") is False, other[src]
    assert ag["provider_preferences"].get("quantizations") == ["fp8"], (other[src], "fp8 must be pinned")
    assert bool(ag.get("text_tools")) == (other[src] == "gemma"), (other[src], "text tools: gemma on, Qwen3.6 off")
    assert isinstance(ag.get("reasoning_effort"), str), other[src]
    assert ag.get("end_task_required") in (None, False) and ag.get("end_confirm") in (None, False) and not ag.get("hack"), (other[src], "the other hero must run plain")
PY
  ) || { echo "merged-config check FAILED" >&2; exit 1; }
  echo "dry run: nothing launched"; exit 0
fi

{ echo "$(date '+%F %T') launching $N_RES resumes from $(basename "$CMDS"), cap \$$CAP"
  echo "  self arm: no override, the checkpoint fleet's own config (gemma-3 27B text tools or Qwen3.6 27B, both DeepInfra fp8)"
  echo "  qwen36 fp8, no full-precision endpoint: $QWEN36"
  echo "  gemma  fp8: $GEMMA"; } >> "$LOG/arms.txt"
echo "$(date '+%F %T') launching $N_RES resumes from $(basename "$CMDS")" >> "$LOG/launch.txt"
# the spend watchdog: stops our resume apps (their description carries --results-dir .../results/raw/hero9_swap/) at the cap
FORENSICS_ENV_FILE="$ROOT/.env" CAP_USD="$CAP" APP_MATCH="/results/raw/$SWAPNAME/" nohup caffeinate -i bash "$HERE/scripts/spend_watchdog.sh" "$SWAPNAME" > "$LOG/watchdog.out" 2>&1 &
echo "$(date '+%F %T') watchdog started (pid $!, cap \$$CAP)" >> "$LOG/launch.txt"
launch_one () {
  log=$1; shift
  [ -f "$log" ] && grep -q "Launching\|Resuming\|Checkpoint:" "$log" && ! grep -q "rate limit exceeded" "$log" && return 0
  for try in 1 2 3 4 5 6 7 8; do
    ( cd "$HERE/agent-interp-envs" && MODAL_SECRETS=${MODAL_SECRETS:-openrouter-secret} nohup env -u VIRTUAL_ENV uv run modal run --detach \
        scripts/resume_modal.py "$@" > "$log" 2>&1 & ); sleep 25
    grep -q "rate limit exceeded" "$log" || break
    sleep $((10 * try))
  done
  echo "$(date '+%F %T') launched $(basename "$log")" >> "$(dirname "$log")/launch.txt"
}
export -f launch_one; export HERE
xargs -P 4 -L 1 bash -c 'eval "launch_one $*"' _ < "$CMDS"
echo "$(date '+%F %T') all launched ($(basename "$CMDS"))" >> "$LOG/launch.txt"
# the fold-in: waits for our apps, fetches, redacts, grades, scores, audits, charts; detached and sleep-safe
spent_now=$(FORENSICS_ENV_FILE="$ROOT/.env" python3 "$HERE/scripts/hero_usage.py" 2>/dev/null | grep "since 18 Oct" | grep -oE '\$[0-9][0-9,]*\.[0-9]+' | head -1 | tr -d '$,')
CEIL=${CEIL:-$(python3 -c "print(round(float('${spent_now:-0}') + float('$CAP') + 100))")} GRADE_CAP=${GRADE_CAP:-60} bash "$HERE/scripts/hero9_swap_foldin.sh"
echo "$(date '+%F %T') fold-in started (ceiling ${CEIL:-see foldin.txt})" >> "$LOG/launch.txt"
