#!/usr/bin/env bash
# Fold in the hero3 swap (6 Nov 2026) unattended and sleep-safe: wait for the launcher ("all launched" in
# results/logs/hero3_swap/launch.txt), then until none of OUR Modal apps (the ap- ids in the launch logs) is running,
# re-checking every 5 min (a closed lid only pauses this). Then fetch (never --force), score, audit, and append the key's
# usage to results/logs/hero3_swap/cost.txt. Every 90 min while apps still run, an interim fetch + score, so partial
# results exist. Gives up waiting after MAX_HOURS awake hours (default 10) and scores what is there. Does NOT commit.
# No grading: all three tasks are scored from their final state.
#   nohup caffeinate -i bash /abs/path/gigi/3-forensics/scripts/hero3_swap_foldin.sh > /abs/path/.../foldin_stdout.txt 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
L="$HERE/results/logs/hero3_swap"; mkdir -p "$L"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
say () { echo "$(date '+%F %T') $*" >> "$L/foldin.txt"; }
MAX=$(( ${MAX_HOURS:-10} * 12 ))
say "start"
until grep -q "all launched" "$L/launch.txt" 2>/dev/null; do sleep 60; done
say "launcher done"
running () {
  (cd "$HERE/agent-interp-envs" && env -u VIRTUAL_ENV uv run python - "$L" <<'PY' 2>> "$L/wait_errors.txt"
import json, pathlib, re, subprocess, sys
ids = set()
for log in pathlib.Path(sys.argv[1]).glob("*__*__*.log"):
    ids |= set(re.findall(r"ap-[A-Za-z0-9]+", log.read_text(errors="replace")))
try:
    r = subprocess.run(["modal", "app", "list", "--json"], capture_output=True, text=True, timeout=120)
    apps = json.loads(r.stdout)
except Exception:
    print("?"); sys.exit()
print(sum(1 for a in apps if (a.get("App ID") or a.get("app_id")) in ids
          and "stopped" not in str(a.get("State", a.get("state", ""))).lower()))
PY
  )
}
fold () {   # fetch + score + audit
  (cd "$HERE" && python3 scripts/hero3_swap_score.py fetch) >> "$L/fetch.txt" 2>&1 || say "fetch had failures (see fetch.txt)"
  (cd "$HERE" && python3 scripts/hero3_swap_score.py) >> "$L/score.txt" 2>&1 && say "scored: $(tail -1 "$L/score.txt")" || say "SCORE FAILED"
}
i=0
while :; do
  n=$(running); i=$((i + 1))
  [ "$n" = 0 ] && { say "all our apps finished"; break; }
  [ $((i % 6)) = 1 ] && say "waiting: ${n:-?} of our apps still running"
  [ $((i % 18)) = 0 ] && { say "interim fold-in"; fold; }
  [ $i -ge $MAX ] && { say "gave up waiting after ${MAX_HOURS:-10} h; scoring what is there"; break; }
  sleep 300
done
fold
(cd "$ROOT" && python3 gigi/3-forensics/scripts/hero_usage.py 2>&1 | grep -E "key usage|since 18 Oct" | sed "s/^/$(date '+%F %T') after: /") >> "$L/cost.txt"
say "done (not committed: the parent session commits)"
