#!/usr/bin/env bash
# Fold in the symmetric hero swap (10 Nov 2026; launched by scripts/hero9_swap.sh) unattended and sleep-safe, modelled on
# hero3_swap_foldin.sh / hero2_swap_foldin.sh. One detached process (nohup caffeinate -i; every step logged to
# results/logs/hero9_swap/foldin.txt; rerun it any time, every step skips what is done):
#   1. wait  for "all launched" in results/logs/hero9_swap/launch.txt, then until none of OUR Modal apps (the ap- ids in the
#            launch logs) is running (re-checked every 5 min; a closed lid only pauses this), then for our local launchers;
#   2. fetch every fleet named in the launch logs once (hero9_swap_score.py fetch: safe_fetch, never --force);
#   3. redact the key (scripts/redact_key.py over results/raw/hero9_swap/);
#   4. grade  the judged tasks' continuations with grade_modal.py (HERO2_SWAP=1 SWAP_HERO_NAME=hero9_swap: Opus; estimate
#            first; stops before a task whose estimate would cross GRADE_CAP or CEIL);
#   5. score  hero9_swap_score.py -> results/hero9_swap.csv, _per_run.csv, _loss_audit.md;
#   6. chart  hero9_swap_chart.py -> charts/hero9_swap_gaps.png, results/hero9_swap_summary.md;
#   7. cost   the key's usage appended to results/logs/hero9_swap/cost.txt.
# Every 90 min while apps still run: an interim fetch + score, so partial results exist. Gives up waiting after MAX_HOURS
# awake hours (default 14) and folds in what is there. Does NOT commit.
#   CEIL=<$ of the key since 18 Oct> GRADE_CAP=60 bash /abs/.../scripts/hero9_swap_foldin.sh    detaches, returns at once
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
L="$HERE/results/logs/hero9_swap"; RAW="$HERE/results/raw/hero9_swap"; mkdir -p "$L"
export MODAL_PROFILE=${MODAL_PROFILE:-spar} MODAL_SECRET_SUFFIX=${MODAL_SECRET_SUFFIX:--gigi} MODAL_VOLUME_PREFIX=${MODAL_VOLUME_PREFIX:-gigi-forensics-}
: "${CEIL:?set CEIL to the grading ceiling (\$ of the key since 18 Oct 2026)}"; GRADE_CAP=${GRADE_CAP:-60}
say () { echo "$(date '+%F %T') $*" >> "$L/foldin.txt"; }
if [ -z "${HERO9_FOLDIN_CHILD:-}" ]; then
  if [ -f "$L/foldin.pid" ] && kill -0 "$(cat "$L/foldin.pid")" 2>/dev/null; then echo "hero9_swap_foldin: already running (pid $(cat "$L/foldin.pid"))"; exit 0; fi
  HERO9_FOLDIN_CHILD=1 nohup caffeinate -i bash "$0" "$@" >> "$L/foldin_stdout.txt" 2>&1 &
  disown 2>/dev/null; echo "hero9_swap_foldin: detached (pid $!); log $L/foldin.txt"; exit 0
fi
echo $$ > "$L/foldin.pid"; trap 'rm -f "$L/foldin.pid"' EXIT
MAX=$(( ${MAX_HOURS:-14} * 12 ))
say "start (ceiling $CEIL, grade cap $GRADE_CAP)"
until grep -q "all launched" "$L/launch.txt" 2>/dev/null; do sleep 60; done
say "launcher done"
running () {
  (cd "$HERE/agent-interp-envs" && env -u VIRTUAL_ENV uv run python - "$L" <<'PY' 2>> "$L/wait_errors.txt"
import json, pathlib, re, subprocess, sys
ids = set()
for log in pathlib.Path(sys.argv[1]).glob("*__*__*.log"):
    ids |= set(re.findall(r"ap-[A-Za-z0-9]+", log.read_text(errors="replace")))
if not ids:
    print(0); sys.exit()
try:
    apps = json.loads(subprocess.run(["modal", "app", "list", "--json"], capture_output=True, text=True, timeout=120).stdout)
except Exception:
    print("?"); sys.exit()
print(sum(1 for a in apps if (a.get("App ID") or a.get("app_id")) in ids and "stopped" not in str(a.get("State", a.get("state", ""))).lower()))
PY
  )
}
fold () {   # fetch + redact + grade + score + chart
  (cd "$HERE" && python3 scripts/hero9_swap_score.py fetch) >> "$L/fetch.txt" 2>&1 || say "fetch had failures (see fetch.txt)"
  if [ -d "$RAW" ]; then
    (cd "$HERE" && FORENSICS_ENV_FILE="$ROOT/.env" python3 scripts/redact_key.py "$RAW") >> "$L/redact.txt" 2>&1 && REDACTED=1 || { REDACTED=0; say "REDACTION FAILED: grading skipped this pass"; }
    [ "$REDACTED" = 1 ] && (cd "$HERE" && FORENSICS_ENV_FILE="$ROOT/.env" HERO2_SWAP=1 SWAP_HERO_NAME=hero9_swap GRADE_CAP_USD="$GRADE_CAP" TOTAL_CEILING_USD="$CEIL" \
        GRADE_LOG_DIR="$L/grading" python3 scripts/grade_modal.py) >> "$L/grade.txt" 2>&1 || say "grading exited non-zero (see grade.txt)"
  fi
  (cd "$HERE" && python3 scripts/hero9_swap_score.py) >> "$L/score.txt" 2>&1 && say "scored: $(tail -1 "$L/score.txt")" || say "SCORE FAILED"
  (cd "$HERE/agent-interp-envs" && env -u VIRTUAL_ENV uv run --with matplotlib python ../scripts/hero9_swap_chart.py) >> "$L/chart.txt" 2>&1 && say "chart: $(tail -1 "$L/chart.txt")" || say "CHART FAILED"
}
i=0
while :; do
  n=$(running); i=$((i + 1))
  [ "$n" = 0 ] && { say "all our apps finished"; break; }
  [ $((i % 6)) = 1 ] && say "waiting: ${n:-?} of our apps still running"
  [ $((i % 18)) = 0 ] && { say "interim fold-in"; fold; }
  [ $i -ge $MAX ] && { say "gave up waiting after ${MAX_HOURS:-14} h; folding in what is there"; break; }
  sleep 300
done
for i in $(seq 1 30); do   # our local launchers download each fleet's analysis set when its app ends
  procs=$(pgrep -f "resume_modal.py .*results-dir .*/hero9_swap/" | wc -l | tr -d ' '); [ "$procs" = 0 ] && break; sleep 60
done
fold
(cd "$ROOT" && FORENSICS_ENV_FILE="$ROOT/.env" python3 gigi/3-forensics/scripts/hero_usage.py 2>&1 | grep -E "key usage|since 18 Oct" | sed "s/^/$(date '+%F %T') after: /") >> "$L/cost.txt"
say "DONE (not committed: the parent session commits)"
