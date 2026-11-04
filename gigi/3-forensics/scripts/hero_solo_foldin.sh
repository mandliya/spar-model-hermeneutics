#!/usr/bin/env bash
# Fold in Llama 3.3 70B's solo runs (hero target, 4 Nov 2026; hero_models_prep.md section 4) unattended, modelled on
# fix_losses_foldin.sh: wait until the hero launchers are done and none of OUR apps (the app ids in the launch logs of
# results/logs/<stage>/*__llama-3.3-70b.log) is running, fetch each fleet once into results/raw/<stage>/<env>/<Output
# path> (stamped .fetched_after_apps_stopped), redact the key, grade the four judged environments (Opus 5, then Opus 4.8
# for what is still ungraded; stops before an environment whose estimate would cross GRADE_CAP or CEIL), score into
# results/hero_full_runs.csv (+ _per_run.csv), write the loss audit to results/hero_loss_audit.md and progress lines to
# results/logs/hero/progress.txt. Does NOT commit or push (the parent session does). Idempotent: every step skips
# what is done; rerun it any time. Each step is wrapped, so one failure does not stop the unrelated later steps
# (grading needs the redaction to have succeeded: transcripts go to the judge).
#   CEIL=<approved ceiling, $ of the key's spend since 18 Oct> caffeinate -i bash gigi/3-forensics/scripts/hero_solo_foldin.sh
# Env: CEIL (required; same meaning as fix_losses_foldin.sh's CEIL / grade_modal.py's TOTAL_CEILING_USD), GRADE_CAP ($ per
# grading pass, default 60), HERO_STAGES (default hero_smoke,hero_solo,hero_solo_key), HERO_MODEL (default llama-3.3-70b),
# WAIT_MINUTES (default 840 awake minutes per wait), DRY_RUN=1 (no network, no fetch, no redaction, no grading: the
# grader prints its estimate only; scoring and audit go to a scratch folder unless HERO_OUT_DIR / HERO_LOG_DIR are set).
# Never fetches with --force: each fleet goes to a temp folder and only files missing locally are moved in, so local-only
# files (final/grading.json) can never be lost.
# A second invocation for another stage (4 Nov 2026: hero_solo_key2 = ai_rd with the capped key; hero_solo_all = the
# other 21 environments incl. swe_rebench_v2, every judged one graded, all 29 scored) must ADD to the gap rows, not
# replace them: pass HERO_MERGE=1. Then the scoring writes results/hero_full_runs__<stages>.csv (+ _per_run) for the
# requested stages and rebuilds results/hero_full_runs.csv (+ _per_run) over those stages plus every stage already in
# results/hero_full_runs_per_run.csv; the audit likewise writes results/hero_loss_audit__<stages>.md and rebuilds the
# merged results/hero_loss_audit.md. Without HERO_MERGE the unsuffixed files cover the requested stages only (as before).
#   HERO_MERGE=1 HERO_STAGES=hero_solo_key2 CEIL=... caffeinate -i bash gigi/3-forensics/scripts/hero_solo_foldin.sh
#   HERO_MERGE=1 HERO_STAGES=hero_solo_all  CEIL=... caffeinate -i bash gigi/3-forensics/scripts/hero_solo_foldin.sh
# Per model (4 Nov 2026, the second pass "hero2" with the end-of-run confirmation: scripts/hero2_launch_solo.sh, stages
# hero2_llama + hero2_llama_key, hero2_qwen + hero2_qwen_key, hero2_gemma + hero2_gemma_key): HERO_MODEL picks the model
# tag (configs/<env>__<tag>.yaml, launch logs <env>__<tag>.log), HERO_STAGES its stages. When a hero2 stage is requested
# or the model is not llama-3.3-70b, nothing of the first pass is touched: the scoring goes to
# results/<HERO_OUT_PREFIX>.csv (+ _per_run; default hero2_full_runs__<model>, full_runs_table.hero_out_prefix), the audit
# to results/<HERO_AUDIT> (default hero2_loss_audit__<model>.md) and the logs to results/logs/hero2/<model>/ (HERO_LOG_DIR).
# With HERO_MERGE=1 the suffixed files are <prefix>__<stages>.csv and <audit stem>__<stages>.md and the unsuffixed
# per-model files are rebuilt over every stage already in <prefix>_per_run.csv. The default run (Llama, hero_* stages)
# writes exactly the files it wrote before. Driver for all three models: scripts/hero2_driver.sh.
#   HERO_MODEL=qwen3-next-80b HERO_STAGES=hero2_qwen,hero2_qwen_key CEIL=... caffeinate -i bash gigi/3-forensics/scripts/hero_solo_foldin.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ENVS="$HERE/agent-interp-envs"; ROOT="$(cd "$HERE/../.." && pwd)"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
MODEL=${HERO_MODEL:-llama-3.3-70b}; export HERO_MODEL="$MODEL"
case "$MODEL" in
  llama-3.3-70b) DEF_STAGES=hero_smoke,hero_solo,hero_solo_key ;;
  qwen3-next-80b) DEF_STAGES=hero2_qwen,hero2_qwen_key ;;
  gemma-3-27b) DEF_STAGES=hero2_gemma,hero2_gemma_key ;;
  *) DEF_STAGES="" ;;
esac
STAGES=${HERO_STAGES:-$DEF_STAGES}; DRY=${DRY_RUN:-0}
[ -n "$STAGES" ] || { echo "hero_solo_foldin.sh: no default stages for HERO_MODEL=$MODEL; set HERO_STAGES" >&2; exit 2; }
WAIT=${WAIT_MINUTES:-840}; GRADE_CAP=${GRADE_CAP:-60}; MERGE=${HERO_MERGE:-0}; export HERO_MERGE="$MERGE"
# second pass or another model: per-model file names, so the first pass's hero_full_runs*.csv / hero_loss_audit*.md stay as they are
if [ "$MODEL" != llama-3.3-70b ] || [[ ",$STAGES," == *",hero2_"* ]]; then
  HERO2=1; PREFIX=${HERO_OUT_PREFIX:-hero2_full_runs__$MODEL}; AUDIT=${HERO_AUDIT:-hero2_loss_audit__$MODEL.md}; DEF_LOG="$HERE/results/logs/hero2/$MODEL"
  [[ ",$STAGES," == *",hero2_"* ]] || { PREFIX=${HERO_OUT_PREFIX:-hero_full_runs__$MODEL}; AUDIT=${HERO_AUDIT:-hero_loss_audit__$MODEL.md}; }
else
  HERO2=0; PREFIX=${HERO_OUT_PREFIX:-hero_full_runs}; AUDIT=${HERO_AUDIT:-hero_loss_audit.md}; DEF_LOG="$HERE/results/logs/hero"
fi
export HERO_OUT_PREFIX="$PREFIX"
if [ "$DRY" = 1 ]; then
  SCR=$(mktemp -d "${TMPDIR:-/tmp}/hero_foldin_dry.XXXXXX")
  L=${HERO_LOG_DIR:-$SCR/logs}; OUTD=${HERO_OUT_DIR:-$SCR}
else
  : "${CEIL:?set CEIL to the approved spend ceiling (\$ of the key since 18 Oct 2026); there is no default}"
  L=${HERO_LOG_DIR:-$DEF_LOG}; OUTD=${HERO_OUT_DIR:-$HERE/results}
fi
mkdir -p "$L" "$OUTD"
R="env -u VIRTUAL_ENV uv run"
say () { echo "$(date '+%F %T') $*" >> "$L/progress.txt"; [ "$DRY" = 1 ] && echo "$*"; return 0; }
step () { local name=$1; shift; say "step $name"; "$@"; local rc=$?; [ $rc = 0 ] || say "STEP FAILED: $name (exit $rc)"; return $rc; }
say "start (model $MODEL, stages $STAGES, ceiling ${CEIL:-none, dry run}, grade cap $GRADE_CAP, files $PREFIX*.csv + $AUDIT$([ "$MERGE" = 1 ] && echo ", merge into the existing files of this model")$([ "$DRY" = 1 ] && echo ", DRY RUN"))"
for s in ${STAGES//,/ }; do [ -d "$HERE/results/logs/$s" ] || say "stage $s: no launch logs yet (results/logs/$s/): nothing to wait for, fetch, grade or score there this pass"; done

# 1. wait for each stage's launcher, then until none of our apps runs and no local launcher of ours is still downloading
wait_apps () {
  local s i running=1 procs
  for s in ${STAGES//,/ }; do
    [ -d "$HERE/results/logs/$s" ] || { say "stage $s: no launch logs yet, not waited for"; continue; }
    for i in $(seq 1 "$WAIT"); do
      grep -q "all launched" "$HERE/results/logs/$s/launch.txt" 2>/dev/null && break
      [ $((i % 30)) = 1 ] && say "waiting for the $s launcher"
      sleep 60
    done
  done
  for i in $(seq 1 "$WAIT"); do
    running=$(cd "$ENVS" && $R python - "$HERE" "$MODEL" "$STAGES" <<'PY' 2>> "$L/wait_errors.txt"
import json, pathlib, re, subprocess, sys
H, model, stages = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
ids = set()
for s in stages:
    for log in (H / "results" / "logs" / s).glob(f"*__{model}.log"):
        ids |= set(re.findall(r"ap-[A-Za-z0-9]+", log.read_text(errors="replace")))
if not ids:
    print(0); sys.exit()
try:   # 2 min: a sleeping laptop can leave the client hung; the loop just asks again
    r = subprocess.run(["modal", "app", "list", "--json"], capture_output=True, text=True, timeout=120)
    apps = json.loads(r.stdout)
except (subprocess.TimeoutExpired, ValueError):
    print("?"); sys.exit()
print(sum(1 for a in apps if (a.get("App ID") or a.get("app_id")) in ids
          and "stopped" not in str(a.get("State", a.get("state", ""))).lower()))
PY
)
    [ "${running:-?}" = 0 ] && break
    [ $((i % 30)) = 1 ] && say "waiting: ${running:-?} of our apps still running"
    sleep 60
  done
  [ "${running:-?}" = 0 ] || { say "our apps still running (or unknown) after $WAIT min: not fetching this pass"; return 1; }
  # the local launchers (modal run --detach, nohup) download each fleet's analysis set when their app ends: give them 30 min
  for i in $(seq 1 30); do
    procs=$(pgrep -f "run_modal.py --config .*__${MODEL}\.yaml" | wc -l | tr -d ' ')
    [ "$procs" = 0 ] && break
    [ "$i" = 1 ] && say "waiting for $procs local launcher(s) to finish downloading"
    sleep 60
  done
  say "all our apps finished"
}

# 2. fetch every fleet named in the launch logs, once, after the apps stopped; never overwrite a local file
fetch () {
  cd "$ENVS" && $R python -u - "$HERE" "$MODEL" "$STAGES" <<'PY' >> "$L/progress.txt" 2>&1
import pathlib, re, shutil, subprocess, sys, tempfile, time
H, model, stages = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
fails = 0
for s in stages:
    for log in sorted((H / "results" / "logs" / s).glob(f"*__{model}.log")):
        t = log.read_text(errors="replace")
        vol = re.search(r"^Volume: (\S+)", t, re.M); out = re.search(r"^Output: results/(\S+?)/?$", t, re.M)
        if not vol or not out:
            print(f"NOT READY (no Volume/Output line): {s}/{log.name}"); fails += 1; continue
        env, fleet = log.stem.split("__")[0], out.group(1)
        dest = H / "results" / "raw" / s / env / fleet; stamp = dest / ".fetched_after_apps_stopped"
        if stamp.exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        # 60 min and 3 tries per fleet (a sleeping laptop leaves a hung client). Into an empty temp folder, then only
        # files missing here are moved in: `modal volume get --force` onto the run dirs would delete final/grading.json.
        for attempt in range(3):
            tmp = pathlib.Path(tempfile.mkdtemp(prefix="hero_fetch_"))
            try:
                r = subprocess.run(["modal", "volume", "get", vol.group(1), fleet, str(tmp)], capture_output=True, text=True, timeout=3600)
                rc, err = r.returncode, (r.stderr or r.stdout).strip()[-300:]
            except subprocess.TimeoutExpired:
                rc, err = -1, "timeout (60 min)"
            if rc == 0:
                src = tmp / pathlib.PurePosixPath(fleet).name
                src = src if src.is_dir() else tmp
                new = kept = differ = 0
                for f in sorted(src.rglob("*")):
                    if f.is_dir() and not f.is_symlink():
                        continue
                    d = dest / f.relative_to(src)
                    if d.exists() or d.is_symlink():
                        kept += 1
                        differ += (not f.is_symlink() and not d.is_symlink() and d.is_file() and d.stat().st_size != f.stat().st_size)
                        continue
                    d.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(f), str(d)); new += 1
                stamp.write_text(f"{time.strftime('%F %T')} {vol.group(1)}:{fleet}: {new} new files, {kept} already here "
                                 f"({differ} of them differ in size; local copy kept)\n")
            shutil.rmtree(tmp, ignore_errors=True)
            if rc == 0:
                break
        if rc == 0:
            print(f"fetched {s}/{env} {fleet}: {stamp.read_text().strip()}" + (f" (attempt {attempt + 1})" if attempt else ""))
        else:
            fails += 1; print(f"FETCH FAILED {s}/{env} {fleet}: {err}")
sys.exit(1 if fails else 0)
PY
}

# 3. redact the key (ai_rd puts the capped agent key in the agent's shell; any sk-or-v1 key is caught)
redact () {
  local dirs=() s
  for s in ${STAGES//,/ }; do [ -d "$HERE/results/raw/$s" ] && dirs+=("$HERE/results/raw/$s"); done
  [ ${#dirs[@]} = 0 ] && { say "nothing to redact yet"; return 0; }
  FORENSICS_ENV_FILE="$ROOT/.env" python3 "$HERE/scripts/redact_key.py" "${dirs[@]}" >> "$L/progress.txt" 2>&1
}

# 4. grade the judged environments (dashboard_costly_failure, grade_your_own, both rendering engines; every judged
#    environment when the stages include hero_solo_all: grade_modal.py reads HERO_STAGES)
grade () {
  local common=(HERO_SOLO=1 HERO_STAGES="$STAGES" HERO_MODEL="$MODEL" GRADE_LOG_DIR="$L/grading")
  if [ "$DRY" = 1 ]; then
    (cd "$HERE" && env "${common[@]}" GRADE_DRY_RUN=1 python3 scripts/grade_modal.py) >> "$L/progress.txt" 2>&1
    return
  fi
  local rc=0
  (cd "$HERE" && env "${common[@]}" FORENSICS_ENV_FILE="$ROOT/.env" GRADE_CAP_USD="$GRADE_CAP" TOTAL_CEILING_USD="$CEIL" \
    python3 scripts/grade_modal.py) >> "$L/grade.txt" 2>&1 || rc=$?
  (cd "$HERE" && env "${common[@]}" FORENSICS_ENV_FILE="$ROOT/.env" GRADE_CAP_USD="$GRADE_CAP" TOTAL_CEILING_USD="$CEIL" \
    JUDGE_MODEL=claude-opus-4-8 python3 scripts/grade_modal.py) >> "$L/grade.txt" 2>&1 || rc=$?
  grep -h "REACHED\|WOULD BE CROSSED\|estimate total" "$L/grading/grade.txt" 2>/dev/null | tail -4 | while read -r line; do say "grading: $line"; done
  return $rc
}

# 5. score: results/hero_full_runs.csv (one row per environment: k/n, Wilson CI, losses) and hero_full_runs_per_run.csv
score () {
  (cd "$HERE" && HERO_OUT_DIR="$OUTD" HERO_STAGES="$STAGES" HERO_MODEL="$MODEL" HERO_OUT_PREFIX="$PREFIX" python3 scripts/full_runs_table.py --hero) > "$L/score.txt" 2>&1 \
    && say "scored: $(grep -m1 "runs found" "$L/score.txt")"
}

# 6. loss audit: every environment and stage, target, launched, found, scored, unscored and lost by cause. With HERO_MERGE=1
#    the requested stages go to hero_loss_audit__<stages>.md and the merged audit (every stage in hero_full_runs_per_run.csv)
#    is rebuilt as hero_loss_audit.md; without it hero_loss_audit.md covers the requested stages (as before).
audit () {
  python3 - "$HERE" "$OUTD" "$STAGES" "$MODEL" "$MERGE" "$PREFIX" "$AUDIT" <<'PY' >> "$L/progress.txt" 2>&1
import collections, csv, pathlib, re, sys
H, OUT, stages, model, merge = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3].split(","), sys.argv[4], sys.argv[5] == "1"
PREFIX, AUDIT = sys.argv[6], sys.argv[7]
sys.path.insert(0, str(H / "scripts"))
import full_runs_table as T
MODEL_NAME = {"llama-3.3-70b": "Llama 3.3 70B", "qwen3-next-80b": "Qwen3-Next-80B", "gemma-3-27b": "gemma-3-27b"}.get(model, model)
CONFIRM = any(s.startswith("hero2_") for s in stages)
PROVIDER = {"provider failure", "OpenRouter budget stop"}
OTHER = {"crash", "no log", "context window exceeded", "harness: no chess engine"}


def audit(stages, per_run, name):
    rows = list(csv.DictReader(open(per_run))) if per_run.exists() else []
    found, scored, why = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
    for r in rows:
        k = (r["environment"], r["stage"]); found[k] += 1
        if r["outcome"] == "scored": scored[k] += 1
        else: why[k][r["outcome"]] += 1
    launched, fetched, noout = {}, {}, set()
    for s in stages:
        launched.update({(e, s): n for e, n in T.hero_launched(s, model).items()})
        for log in (H / "results" / "logs" / s).glob(f"*__{model}.log"):
            o = re.search(r"^Output: results/(\S+?)/?$", log.read_text(errors="replace"), re.M)
            env = log.stem.split("__")[0]
            fetched[(env, s)] = bool(o) and (H / "results" / "raw" / s / env / o.group(1) / ".fetched_after_apps_stopped").exists()
            if not o: noout.add((env, s))   # the launch died before its runs started (e.g. a missing Modal secret)
    envs = T.hero_envs(stages) if hasattr(T, "hero_envs") else T.HERO_ENVS
    head = "| environment | stage | target | launched | runs found | fetched | scored | unscored, by reason | lost, provider | lost, other (crash, context, no log) | missing (target - found) |"
    prec = sorted({T.hero_precision(e, model) for e in envs}) if hasattr(T, "hero_precision") else ["fp16 (coreweave, via OpenRouter)"]
    out = [f"# Hero loss audit: {MODEL_NAME} solo runs{' with the end-of-run confirmation (agent.end_confirm=true)' if CONFIRM else ''} (written by scripts/hero_solo_foldin.sh)", "",
           f"Model {model}, {' / '.join(prec)} (as pinned in configs/<env>__{model}.yaml). Stages: {', '.join(stages)}.",
           f"One row per environment ({len(envs)}: the 8 gap environments, or all 29 when hero_solo_all / hero2_llama_all is among the stages) and hero",
           "stage: the planned target (hero_models_prep.md section 4; for hero_solo_all what launch.txt says was launched), runs",
           "launched (launch.txt), runs found locally, whether the fleet was fetched after the apps stopped, runs scored, and every",
           f"other run by cause. Rows below target are listed again at the end. Source: results/{per_run.name}.", "", head, "|" + "---|" * 11]
    below, tot = [], collections.Counter()
    for env in envs:
        for s in stages:
            k = (env, s); target = T.hero_target(s, env, model)
            if not target and not found[k] and k not in launched:
                continue
            lp = sum(c for w, c in why[k].items() if w in PROVIDER)
            lo = "; ".join(f"{w}: {c}" for w, c in sorted(why[k].items()) if w in OTHER)
            un = "; ".join(f"{w}: {c}" for w, c in sorted(why[k].items()) if w not in PROVIDER | OTHER)
            row = (f"| {env} | {s} | {target} | {launched.get(k, 'not launched')}{' (log has no Output line: launch failed?)' if k in noout else ''} | {found[k]} | {'yes' if fetched.get(k) else 'no'} | "
                   f"{scored[k]} | {un} | {lp or ''} | {lo} | {max(0, target - found[k]) or ''} |")
            out.append(row); tot["target"] += target; tot["found"] += found[k]; tot["scored"] += scored[k]
            if scored[k] < target: below.append(row)
    out += ["", f"Total: target {tot['target']}, found {tot['found']}, scored {tot['scored']}.", "",
            f"## Below target: {len(below)} cells", "", head, "|" + "---|" * 11] + below
    (OUT / name).write_text("\n".join(out) + "\n")
    print(f"audit ({', '.join(stages)}): target {tot['target']}, found {tot['found']}, scored {tot['scored']}, {len(below)} cells below target -> {OUT / name}")


stem = AUDIT[:-3] if AUDIT.endswith(".md") else AUDIT
if merge:
    suffix = "__" + "+".join(stages)
    audit(stages, OUT / f"{PREFIX}_per_run{suffix}.csv", f"{stem}{suffix}.md")
    merged_per_run = OUT / f"{PREFIX}_per_run.csv"
    merged = list(dict.fromkeys(stages + [r["stage"] for r in csv.DictReader(open(merged_per_run))])) if merged_per_run.exists() else stages
    audit(merged, merged_per_run, AUDIT)
else:
    audit(stages, OUT / f"{PREFIX}_per_run.csv", AUDIT)
PY
}

APPS_DONE=0; REDACTED=0
if [ "$DRY" = 1 ]; then
  say "dry run: not waiting, not fetching, not redacting"
else
  step wait wait_apps && APPS_DONE=1
  if [ $APPS_DONE = 1 ]; then step fetch fetch; else say "fetch skipped: our apps have not all stopped"; fi
  step redact redact && REDACTED=1
fi
if [ "$DRY" = 1 ] || [ $REDACTED = 1 ]; then step grade grade; else say "grading skipped: redaction did not succeed"; fi
step score score
step audit audit && say "audit written: $OUTD/$AUDIT"
say "done (not committed: the parent session commits)"
[ "$DRY" = 1 ] && echo "dry-run outputs in $OUTD and $L"
exit 0
