#!/usr/bin/env bash
# hero9 supervisor (10 Nov 2026; the call dropped Llama, the heroes are Qwen3.6 27B and gemma-3 27B in the PLAIN
# environment, no check): drives the Qwen3.6 fleet from the smoke runs to the scored results with nobody at the laptop.
# Every wait is a loop, so a sleeping Mac only pauses it. Modelled on scripts/hero8_until_done.sh.
#   1. waits until the smoke apps (results/logs/hero9_smoke_<tag>/*.log, launched by hero9_launch.sh smoke) stopped;
#   2. fetches each smoke fleet into results/raw/hero9_smoke_<tag>/<env>/<fleet> (never --force, never overwrites);
#   3. checks each smoke run's rollout.log the way a NO-check run ends: the harness printed its FINAL RESULTS block (the
#      submit ending in coding, the game end in tictactoe, the plain-reply ending in eval_tampering) and the run is
#      "finished" by full_runs_table.status (no provider failure, no crash). All three pass -> the model goes on;
#   4. launches the full fleets for the passing models (hero9_launch.sh full <model>, skips what is launched already);
#   5. runs hero9_foldin.sh (wait, fetch, redact, grade within CEIL / GRADE_CAP, score, audit) for them;
#   6. writes DONE. Log: results/logs/hero9/until_done.txt. Does not commit (the parent session does).
#   CEIL=<$ since 18 Oct> MODELS="qwen3.6-27b" nohup caffeinate -i bash /abs/.../scripts/hero9_until_done.sh > /abs/log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
: "${CEIL:?set CEIL}"; MODELS=${MODELS:-"qwen3.6-27b"}
L="$HERE/results/logs/hero9"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" | tee -a "$L/until_done.txt"; }
R="env -u VIRTUAL_ENV uv run"
say "start (ceiling $CEIL, models: $MODELS)"
TAGS=""; for m in $MODELS; do t=qwen36; [ "$m" = gemma-3-27b ] && t=gemma; TAGS="$TAGS hero9_smoke_$t"; done

# 1. wait for the smoke apps
until grep -q "all launched (hero9 smoke" "$L/launch.txt" 2>/dev/null; do sleep 60; done
for i in $(seq 1 240); do   # up to 4 h
  running=$(cd agent-interp-envs && $R python - "$HERE" $TAGS <<'PY'
import json, pathlib, re, subprocess, sys
H = pathlib.Path(sys.argv[1]); ids = set()
for s in sys.argv[2:]:
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
for m in $MODELS; do
  tag=qwen36; [ "$m" = gemma-3-27b ] && tag=gemma
  verdict=$(cd agent-interp-envs && $R python - "$HERE" "hero9_smoke_$tag" "$m" <<'PY'
import pathlib, re, shutil, subprocess, sys, tempfile
H, stage, model = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path.insert(0, str(H / "scripts")); import full_runs_table as T
ok, notes = True, []
for log in sorted((H / "results" / "logs" / stage).glob(f"*__{model}.log")):
    t = log.read_text(errors="replace"); env = log.stem.split("__")[0]
    vol = re.search(r"^Volume: (\S+)", t, re.M); out = re.search(r"^Output: results/(\S+?)/?$", t, re.M)
    if not (vol and out):
        ok = False; notes.append(f"{env}: no Volume/Output line in the launch log"); continue
    fleet = out.group(1); dest = H / "results" / "raw" / stage / env / fleet
    if not any(dest.glob("run-*/rollout.log")):
        dest.mkdir(parents=True, exist_ok=True)
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="hero9_smoke_"))
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
    run = logs[0].parent; r = logs[0].read_text(errors="replace")
    st = T.status(run); final = "FINAL RESULTS" in r
    steps = len(list(run.glob("step-*")))
    capped = re.search(r"max steps reached|Reached max steps", r, re.I) is not None
    if st == "finished" and final:
        notes.append(f"{env}: OK (status {st}, FINAL RESULTS printed, {steps} steps{', ended at the step cap' if capped else ''})")
    else:
        ok = False; notes.append(f"{env}: NO proper ending (status {st}, FINAL RESULTS {'present' if final else 'missing'}, {steps} steps)")
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
  bash "$HERE/scripts/hero9_launch.sh" full "$m" >> "$L/launch_full.out" 2>&1
  say "launcher exit $? for $m"
done

# 5. fold in (disk guard: the Mac hit 0 bytes free on 10 Nov; the fold-in fetches tens of GB)
free_gb=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
if [ "${free_gb:-0}" -lt 5 ]; then say "only ${free_gb} GB free on disk: NOT fetching; free space and rerun hero9_foldin.sh by hand"; exit 1; fi
say "starting fold-in for:$PASS (${free_gb} GB free)"
CEIL="$CEIL" MODELS="${PASS# }" bash "$HERE/scripts/hero9_foldin.sh" >> "$L/foldin.out" 2>&1
say "fold-in exit $?"
say "DONE"
