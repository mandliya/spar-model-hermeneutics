#!/usr/bin/env bash
# hero8 supervisor (9 Nov 2026 evening, GG: "ALL DETACHED NOW cuz I need to go"): drives the stand-in fleet from the
# smoke runs to the scored results with nobody at the laptop. Resumes after a sleep (every wait is a loop).
#   1. waits until the smoke apps (results/logs/hero8_smoke_<tag>/*.log, launched by hero8_launch.sh smoke both) stopped;
#   2. fetches each smoke fleet into results/raw/hero8_smoke_<tag>/<env>/<fleet> (never --force, never overwrites);
#   3. checks each smoke run's rollout.log: a parsed finish-tool call followed by "Session ended", or the game's
#      "FINAL RESULTS", or a submission; one smoke per loop type (coding = own finish tool, tictactoe = game loop,
#      eval_tampering = plain-reply ending). A model whose three smokes all pass goes on; otherwise SMOKE FAILED;
#   4. launches the full fleets for the passing models (hero8_launch.sh full <model>, skips what is launched already);
#   5. runs hero8_foldin.sh (wait, fetch, redact, grade within CEIL / GRADE_CAP, score, audit) for them;
#   6. writes DONE. Log: results/logs/hero8/until_done.txt. Does not commit (the parent session does).
#   CEIL=<$ since 18 Oct> nohup caffeinate -i bash /abs/.../scripts/hero8_until_done.sh > /abs/log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
: "${CEIL:?set CEIL}"
L="$HERE/results/logs/hero8"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" | tee -a "$L/until_done.txt"; }
R="env -u VIRTUAL_ENV uv run"
say "start (ceiling $CEIL)"

# 1. wait for the smoke apps
until grep -q "all launched (hero8 smoke" "$L/launch.txt" 2>/dev/null; do sleep 60; done
for i in $(seq 1 240); do   # up to 4 h (eval_tampering 20 steps, coding 100 steps, tictactoe: minutes each)
  running=$(cd agent-interp-envs && $R python - "$HERE" <<'PY'
import json, pathlib, re, subprocess, sys
H = pathlib.Path(sys.argv[1]); ids = set()
for s in ("hero8_smoke_gemma", "hero8_smoke_qwen36"):
    for log in (H / "results" / "logs" / s).glob("*.log"):
        ids |= set(re.findall(r"ap-[A-Za-z0-9]+", log.read_text(errors="replace")))
try:
    apps = json.loads(subprocess.run(["modal", "app", "list", "--json"], capture_output=True, text=True, timeout=120).stdout)
except Exception:
    print("?"); sys.exit()
print(sum(1 for a in apps if (a.get("App ID") or a.get("app_id")) in ids and "stopped" not in str(a.get("State", a.get("state", ""))).lower()))
PY
)
  [ "${running:-?}" = 0 ] && break
  [ $((i % 10)) = 1 ] && say "smoke: ${running:-?} apps still running"
  sleep 60
done
[ "${running:-?}" = 0 ] || { say "smoke apps still running after 4 h: giving up, nothing launched"; exit 1; }
sleep 120   # let the local launchers finish their own download
say "smoke apps finished"

# 2 + 3. fetch and check
PASS=""
for m in gemma-3-27b qwen3.6-27b; do
  tag=gemma; [ "$m" = qwen3.6-27b ] && tag=qwen36
  verdict=$(cd agent-interp-envs && $R python - "$HERE" "hero8_smoke_$tag" "$m" <<'PY'
import pathlib, re, shutil, subprocess, sys, tempfile
H, stage, model = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
ok, notes = True, []
for log in sorted((H / "results" / "logs" / stage).glob(f"*__{model}.log")):
    t = log.read_text(errors="replace"); env = log.stem.split("__")[0]
    vol = re.search(r"^Volume: (\S+)", t, re.M); out = re.search(r"^Output: results/(\S+?)/?$", t, re.M)
    if not (vol and out):
        ok = False; notes.append(f"{env}: no Volume/Output line in the launch log"); continue
    fleet = out.group(1); dest = H / "results" / "raw" / stage / env / fleet
    if not any(dest.glob("run-*/rollout.log")):
        dest.mkdir(parents=True, exist_ok=True)
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="hero8_smoke_"))
        r = subprocess.run(["modal", "volume", "get", vol.group(1), fleet, str(tmp)], capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            ok = False; notes.append(f"{env}: fetch failed: {(r.stderr or r.stdout).strip()[-200:]}"); continue
        src = tmp / pathlib.PurePosixPath(fleet).name; src = src if src.is_dir() else tmp
        for f in src.rglob("*"):
            if f.is_file():
                d = dest / f.relative_to(src)
                if not d.exists():
                    d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, d)
        shutil.rmtree(tmp, ignore_errors=True)
    logs = list(dest.glob("run-*/rollout.log"))
    if not logs:
        ok = False; notes.append(f"{env}: no rollout.log fetched"); continue
    r = logs[0].read_text(errors="replace")
    finish = re.search(r"Function: (end_task|done|end_session)\b", r) and ("Session ended" in r or "submitted" in r.lower() or "FINAL RESULTS" in r or "Game Over" in r or "done" in r)
    game = "FINAL RESULTS" in r or "Game Over" in r
    submitted = re.search(r"submit_solution\.py|Submission (received|successful)|submitted", r, re.I) is not None
    capped = re.search(r"max.?steps reached|hit max steps|Timeout|timed out", r, re.I) is not None
    reminders = r.count("A reply without a tool call does not end the session")
    if finish or game or submitted:
        notes.append(f"{env}: OK ({'finish tool' if finish else 'game end' if game else 'submitted'}; {reminders} reminders)")
    else:
        ok = False; notes.append(f"{env}: NO proper ending ({'step cap/timeout' if capped else 'no finish call parsed'}; {reminders} reminders)")
print("PASS" if ok else "FAIL"); print("\n".join(notes))
PY
)
  say "smoke $m: $(echo "$verdict" | tr '\n' ' | ')"
  echo "$verdict" | head -1 | grep -q PASS && PASS="$PASS $m" || say "SMOKE FAILED $m: full fleet NOT launched"
done
[ -n "$PASS" ] || { say "no model passed the smoke: nothing launched"; exit 1; }

# 4. full fleets for the passing models
for m in $PASS; do
  say "launching full fleet for $m"
  bash "$HERE/scripts/hero8_launch.sh" full "$m" >> "$L/launch_full.out" 2>&1
  say "launcher exit $? for $m"
done

# 5. fold in
say "starting fold-in for:$PASS"
CEIL="$CEIL" MODELS="${PASS# }" bash "$HERE/scripts/hero8_foldin.sh" >> "$L/foldin.out" 2>&1
say "fold-in exit $?"
say "DONE"
