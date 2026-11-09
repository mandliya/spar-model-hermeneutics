#!/usr/bin/env bash
# Full runs on Modal: for one model, N runs per environment at the paper's own step caps, every
# environment launched as one detached app (its N runs start together), launches spaced GAP seconds
# with retries when Modal's app-creation limit turns one away. The laptop must stay awake until the
# last launch has started its runs; after that the runs carry on without it.
#   bash gigi/3-forensics/scripts/modal_full.sh <stage> <model-tag> <env:count> ...
#   e.g. modal_full.sh full_qwen qwen3-next-80b tictactoe:20 puppeteer:5
# Environments whose shipped config has no `modal:` block (no timing bench, no pinned machine) run
# with 2 GB of memory instead of run_modal.py's 8 GB default: host-side only, cheaper, invisible to
# the agent. Per-run wall clock: 1 h for caps <= 60 steps, else 4 h. Results stay on Modal's volumes
# (scripts/modal_sync.py); one log per environment in results/logs/<stage>/.
# Shared workspace (SPAR): MODAL_PROFILE, MODAL_SECRET_SUFFIX and MODAL_VOLUME_PREFIX come from the environment
# (run_modal.py reads the last two); MAXC caps how many of an environment's runs go at once (default 5).
set -uo pipefail
STAGE=$1; MODEL=$2; shift 2
HERE="$(cd "$(dirname "$0")/.." && pwd)"
GAP=${GAP:-20}
LOG="$HERE/results/logs/$STAGE"; mkdir -p "$LOG"
cd "$HERE/agent-interp-envs"
for spec in "$@"; do
  e=${spec%%:*}; n=${spec##*:}
  cfg="$HERE/configs/${e}__${MODEL}.yaml"; f="$LOG/${e}__${MODEL}.log"
  # launched already: skip, unless the launch was turned away (rate/spend limit) or the laptop lost the network before the
  # app completed (9 Nov 2026: the Mac slept mid-fleet, `modal run --detach` keeps only the last function alive, the log
  # ends in ConnectionError / "Could not connect to the Modal server" and no "App completed"): then launch again.
  if [ -f "$f" ] && ! grep -qE "rate limit exceeded|exceeded its spend limit" "$f"; then
    if { tail -n 40 "$f" | grep -qE "ConnectionError|nodename nor servname|Could not connect to the Modal server" && ! grep -q "App completed" "$f"; } || grep -q "launch failed: ConflictError" "$f"; then   # tail only: a transcript in the log may mention ConnectionError while the app is alive
      mkdir -p "$LOG/dead"; mv "$f" "$LOG/dead/$(basename "$f" .log).$(date +%H%M%S).log"   # keep the dead log (its Output line names a partial fleet on the volume)
    else
      continue
    fi
  fi
  read cap hasmodal < <(env -u VIRTUAL_ENV uv run python -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); print(c['agent']['max_steps'], int('modal' in c))" "$cfg")
  ov="${EXTRA_OV:-}"; [ "$hasmodal" = 0 ] && ov="${ov:+$ov,}modal.memory=2048"   # EXTRA_OV: more overrides, e.g. agent.max_steps=30
  to=14400; [ "$cap" -le 60 ] && to=3600
  for try in 1 2 3 4 5 6; do
    MODAL_SECRETS=${MODAL_SECRETS:-openrouter-secret} nohup env -u VIRTUAL_ENV uv run modal run --detach scripts/run_modal.py --config "$cfg" \
      --count "$n" --max-concurrent "${MAXC:-5}" ${ov:+--overrides "$ov"} --timeout "$to" --no-judge \
      --results-dir "$HERE/results/raw/$STAGE/$e" > "$f" 2>&1 &
    pid=$!; sleep "$GAP"
    grep -q "rate limit exceeded" "$f" || break
    echo "$(date '+%F %T') ${e}: rate-limited, retry $try" >> "$LOG/launch.txt"; sleep $((GAP * try))
  done
  echo "$(date '+%F %T') launched ${e}__${MODEL} x$n (max_steps $cap, timeout ${to}s${ov:+, $ov}, pid $pid)" >> "$LOG/launch.txt"
done
echo "$(date '+%F %T') all launched" >> "$LOG/launch.txt"
