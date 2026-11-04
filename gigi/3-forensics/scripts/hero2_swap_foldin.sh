#!/usr/bin/env bash
# Fold in the hero2 swap (4 Nov 2026: scripts/hero_swap.sh HERO2=1, launched by hero_swap_when_ready.sh HERO2=1 from
# hero2_driver.sh; arms llama = control, qwen36, gemma, all with the end-of-run confirmation) unattended, modelled on
# hero_solo_foldin.sh. One detached, sleep-safe process (nohup caffeinate -i; every step logged to
# results/logs/hero2_swap/foldin.txt; no questions; rerun it any time: every step skips what is done):
#   1. wait  until the swap is launched (results/logs/hero2_swap/launch.txt says "all launched"; up to LAUNCH_WAIT_MINUTES,
#            default 2880), then until none of OUR apps (the ap-... ids in results/logs/hero2_swap/*.log, top-ups included)
#            is running (up to WAIT_MINUTES awake minutes, default 840), then up to 30 min for our local launchers to finish
#            their downloads;
#   2. fetch every fleet named in the launch logs once, after the apps stopped (stamp .fetched_after_apps_stopped), with
#            swap_all_score.safe_fetch: `modal volume get` into an empty temp folder, then only the files missing locally
#            are moved in. NEVER `modal volume get --force` onto a run folder (that would delete a local final/grading.json);
#   3. redact the key (scripts/redact_key.py over results/raw/hero2_swap/);
#   4. grade  the judged environments' replays (dashboard_costly_failure, grade_your_own, optimize_rendering_engine,
#            optimize_rendering_engine_graded) with grade_modal.py HERO2_SWAP=1: Opus 5, then Opus 4.8 for what is still
#            ungraded; estimate first; stops before an environment whose estimate would cross GRADE_CAP or CEIL;
#   5. score  with the HERO2 modes: swap_all_score.py --hero2 (results/hero2_swap.csv), swap_validity.py --hero2
#            (results/hero2_swap_validity.csv), swap_endings.py --hero2 (results/hero2_swap_endings.csv), gap_replays.py --hero2
#            (results/hero2_gap_replays.csv). label_endings_llm.py (Claude Haiku, paid) is NOT run here;
#   6. audit  results/hero2_swap_loss_audit.md: every restart point x arm: target, scored, cheated, lost by cause;
#   7. chart  scripts/hero2_gap_chart.py -> charts/hero2_gap.png (+ its markdown table in results/logs/hero2_swap/chart.txt).
# Each step is wrapped, so one failure does not stop the unrelated later steps (grading needs the redaction to have
# succeeded: transcripts go to the judge). Nothing under results/ is deleted; the files above are this script's own
# outputs, rewritten on each pass. Does NOT commit or push (the parent session does).
#   CEIL=<approved ceiling, $ of the key's spend since 18 Oct 2026> bash gigi/3-forensics/scripts/hero2_swap_foldin.sh   detaches, returns at once
#   DRY_RUN=1 bash gigi/3-forensics/scripts/hero2_swap_foldin.sh   foreground, no network, no Modal, no fetch, no redaction, no
#            grading (the grader prints its estimate only); scoring, audit and chart go to a scratch folder
# Env: CEIL (required unless DRY_RUN=1; grade_modal.py's TOTAL_CEILING_USD), GRADE_CAP ($ per grading pass, default 60),
# WAIT_MINUTES (default 840), LAUNCH_WAIT_MINUTES (default 2880), HERO2_LOG_DIR / HERO2_OUT_DIR (dry runs: where the logs /
# the outputs go), MODAL_PROFILE / MODAL_SECRET_SUFFIX / MODAL_VOLUME_PREFIX (default the SPAR workspace values).
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ENVS="$HERE/agent-interp-envs"; ROOT="$(cd "$HERE/../.." && pwd)"
export MODAL_PROFILE=${MODAL_PROFILE:-spar} MODAL_SECRET_SUFFIX=${MODAL_SECRET_SUFFIX:--gigi} MODAL_VOLUME_PREFIX=${MODAL_VOLUME_PREFIX:-gigi-forensics-}
DRY=${DRY_RUN:-0}; WAIT=${WAIT_MINUTES:-840}; LWAIT=${LAUNCH_WAIT_MINUTES:-2880}; GRADE_CAP=${GRADE_CAP:-60}
SWAPLOG="$HERE/results/logs/hero2_swap"; RAW="$HERE/results/raw/hero2_swap"; PLAN="$HERE/results/hero2_swap_plan.json"
export SWAP_HERO2=1
if [ "$DRY" = 1 ]; then
  SCR=$(mktemp -d "${TMPDIR:-/tmp}/hero2_swap_foldin_dry.XXXXXX")
  L=${HERO2_LOG_DIR:-$SCR/logs}; OUTD=${HERO2_OUT_DIR:-$SCR/results}; CHART="$SCR/charts/hero2_gap.png"
  # the scoring scripts read the real raw/, logs/ and plan through a stand-in results/ folder and write their outputs there
  # (a HERO2_OUT_DIR that already holds raw/, logs/ or a plan is used as it is: a test fixture)
  mkdir -p "$OUTD"; [ -e "$OUTD/raw" ] || ln -s "$HERE/results/raw" "$OUTD/raw"; [ -e "$OUTD/logs" ] || ln -s "$HERE/results/logs" "$OUTD/logs"
  [ -f "$PLAN" ] && [ ! -e "$OUTD/hero2_swap_plan.json" ] && ln -s "$PLAN" "$OUTD/hero2_swap_plan.json"
  export SWAP_HERO2_RESULTS="$OUTD" HERO2_CHART_OUT="$CHART"
else
  : "${CEIL:?set CEIL to the approved spend ceiling (\$ of the key since 18 Oct 2026); there is no default}"
  L=${HERO2_LOG_DIR:-$SWAPLOG}; OUTD="$HERE/results"; CHART="$HERE/charts/hero2_gap.png"
fi
mkdir -p "$L"
R="env -u VIRTUAL_ENV uv run"
say () { echo "$(date '+%F %T') $*" >> "$L/foldin.txt"; [ "$DRY" = 1 ] && echo "$*"; return 0; }
step () { local name=$1; shift; say "step $name"; "$@"; local rc=$?; [ $rc = 0 ] || say "STEP FAILED: $name (exit $rc)"; return $rc; }

# ---- detach (real runs only): the child does the work under caffeinate, this shell returns at once ----
if [ "$DRY" != 1 ] && [ -z "${HERO2_FOLDIN_CHILD:-}" ]; then
  if [ -f "$L/foldin.pid" ] && kill -0 "$(cat "$L/foldin.pid")" 2>/dev/null; then
    echo "hero2_swap_foldin: already running (pid $(cat "$L/foldin.pid")); log $L/foldin.txt"; exit 0
  fi
  HERO2_FOLDIN_CHILD=1 nohup caffeinate -i bash "$0" "$@" >> "$L/foldin_stdout.txt" 2>&1 &
  disown 2>/dev/null; echo "hero2_swap_foldin: detached (pid $!); log $L/foldin.txt"; exit 0
fi
if [ "$DRY" != 1 ]; then
  if [ -f "$L/foldin.pid" ] && [ "$(cat "$L/foldin.pid")" != $$ ] && kill -0 "$(cat "$L/foldin.pid")" 2>/dev/null; then
    say "another fold-in is running (pid $(cat "$L/foldin.pid")): exiting"; exit 0
  fi
  echo $$ > "$L/foldin.pid"; trap 'rm -f "$L/foldin.pid"' EXIT
fi
say "start (ceiling ${CEIL:-none, dry run}, grade cap $GRADE_CAP, raw $RAW$([ "$DRY" = 1 ] && echo ", DRY RUN: outputs in $SCR"))"
[ -d "$RAW" ] || say "note: $RAW does not exist yet (no hero2 swap replay fetched)"
[ -f "$PLAN" ] || say "note: $PLAN does not exist yet (hero_swap_when_ready.sh HERO2=1 writes it)"

# 1. wait: the launch, then our apps, then our local launchers
wait_apps () {
  local i running procs
  for i in $(seq 1 "$LWAIT"); do
    grep -q "all launched" "$SWAPLOG/launch.txt" 2>/dev/null && break
    [ $((i % 30)) = 1 ] && say "waiting for the swap to be launched (no 'all launched' in results/logs/hero2_swap/launch.txt yet; chain: $(tail -1 "$SWAPLOG/chain.txt" 2>/dev/null | cut -c21-200))"
    sleep 60
  done
  grep -q "all launched" "$SWAPLOG/launch.txt" 2>/dev/null || { say "the swap is not launched after $LWAIT min: nothing to fetch this pass"; return 1; }
  for i in $(seq 1 "$WAIT"); do
    running=$(cd "$ENVS" && $R python - "$SWAPLOG" <<'PY' 2>> "$L/wait_errors.txt"
import json, pathlib, re, subprocess, sys
ids = set()
for log in pathlib.Path(sys.argv[1]).glob("*__*__*.log"):   # every launch log, top-ups included
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
  for i in $(seq 1 30); do   # the launchers (modal run --detach) download each fleet's analysis set when its app ends
    procs=$(pgrep -f "resume_modal.py .*results-dir .*/hero2_swap/" | wc -l | tr -d ' ')
    [ "$procs" = 0 ] && break
    [ "$i" = 1 ] && say "waiting for $procs local launcher(s) to finish downloading"
    sleep 60
  done
  say "all our apps finished"
}

# 2. fetch every fleet named in the launch logs, once, after the apps stopped; never --force, never over a local file
fetch () {
  python3 -u - "$HERE" "$SWAPLOG" "$RAW" <<'PY' >> "$L/foldin.txt" 2>&1
import pathlib, re, sys, time
H, LOGS, RAW = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
sys.path.insert(0, str(H / "scripts"))
import swap_all_score as S   # SWAP_HERO2=1 in the environment: S.safe_fetch, S.FLEET_RE
fails = done = 0
for log in sorted(LOGS.glob("*__*__*.log")):
    env, point, who = log.stem.split("__")[:3]
    t = log.read_text(errors="replace"); vol = re.search(r"^Volume: (\S+)", t, re.M)
    fleets = sorted({m[0] for m in S.FLEET_RE.findall(t)})
    if not vol or not fleets:
        print(f"NOT READY (no Volume line or no fleet yet): {log.name}"); fails += 1; continue
    for fleet in fleets:
        dest = RAW / env / point / who / fleet; stamp = dest / ".fetched_after_apps_stopped"
        if stamp.exists():
            continue
        for attempt in range(3):   # 60 min and 3 tries per fleet (a sleeping laptop leaves a hung client)
            rc, new, kept, differ, err = S.safe_fetch(vol.group(1), fleet, dest)
            if rc == 0:
                stamp.write_text(f"{time.strftime('%F %T')} {vol.group(1)}:{fleet}: {new} new files, {kept} already here "
                                 f"({differ} of them differ in size; local copy kept)\n")
                break
        if rc == 0:
            done += 1; print(f"fetched {env} {point} {who} {fleet}: {stamp.read_text().strip()}" + (f" (attempt {attempt + 1})" if attempt else ""))
        else:
            fails += 1; print(f"FETCH FAILED {env} {point} {who} {fleet}: {err}")
print(f"fetch: {done} fleets fetched, {fails} failed or not ready")
sys.exit(1 if fails else 0)
PY
}

# 3. redact the key (ai_rd puts the capped agent key in the agent's shell; any sk-or-v1 key is caught)
redact () {
  [ -d "$RAW" ] || { say "nothing to redact yet"; return 0; }
  FORENSICS_ENV_FILE="$ROOT/.env" python3 "$HERE/scripts/redact_key.py" "$RAW" >> "$L/foldin.txt" 2>&1
}

# 4. grade the judged environments' replays
grade () {
  local common=(HERO2_SWAP=1 GRADE_LOG_DIR="$L/grading")
  if [ "$DRY" = 1 ]; then
    (cd "$HERE" && env "${common[@]}" GRADE_DRY_RUN=1 python3 scripts/grade_modal.py) >> "$L/foldin.txt" 2>&1
    return
  fi
  local rc=0
  (cd "$HERE" && env "${common[@]}" FORENSICS_ENV_FILE="$ROOT/.env" GRADE_CAP_USD="$GRADE_CAP" TOTAL_CEILING_USD="$CEIL" \
    python3 scripts/grade_modal.py) >> "$L/grade.txt" 2>&1 || rc=$?
  (cd "$HERE" && env "${common[@]}" FORENSICS_ENV_FILE="$ROOT/.env" GRADE_CAP_USD="$GRADE_CAP" TOTAL_CEILING_USD="$CEIL" \
    JUDGE_MODEL=claude-opus-4-8 python3 scripts/grade_modal.py) >> "$L/grade.txt" 2>&1 || rc=$?
  grep -h "REACHED\|WOULD BE CROSSED\|estimate total\|not yet" "$L/grading/grade.txt" 2>/dev/null | tail -4 | while read -r line; do say "grading: $line"; done
  return $rc
}

# 5. score with the HERO2 modes (order matters: validity -> endings -> gap_replays)
score () {
  local rc=0
  (cd "$ENVS" && $R python ../scripts/swap_all_score.py --hero2) >> "$L/foldin.txt" 2>&1 || rc=$?
  (cd "$ENVS" && $R python ../scripts/swap_validity.py --hero2) > "$L/swap_validity.txt" 2>&1 || rc=$?
  head -1 "$L/swap_validity.txt" >> "$L/foldin.txt"
  (cd "$ENVS" && $R python ../scripts/swap_endings.py --hero2) > "$L/swap_endings.txt" 2>&1 || rc=$?
  tail -1 "$L/swap_endings.txt" >> "$L/foldin.txt"
  (cd "$ENVS" && $R python ../scripts/gap_replays.py --hero2) >> "$L/foldin.txt" 2>&1 || rc=$?
  say "scoring done (label_endings_llm.py not run: it spends; run it by hand if the 'other' endings matter)"
  return $rc
}

# 6. the loss audit: every restart point x arm
audit () {
  python3 - "$HERE" "$OUTD" <<'PY' >> "$L/foldin.txt" 2>&1
import collections, csv, pathlib, sys
H, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sys.path.insert(0, str(H / "scripts"))
import swap_all_score as S   # HERO2 mode (SWAP_HERO2=1): S.PLAN, S.WHO, S.RES
if S.not_ready():
    print(f"audit: {S.not_ready()}"); sys.exit(0)
gap = S.RES / "hero2_gap_replays.csv"
if not gap.exists():
    print(f"audit: {gap.name} not written yet (gap_replays.py --hero2): no audit"); sys.exit(0)
UNFIX = {"context window exceeded", "no measure", "no value for the measure", "step cap, no report"}
ARM = {"llama": "Llama (control)", "qwen36": "Qwen3.6", "gemma": "gemma-3", "gemma4": "gemma 4", "gptoss": "GPT-OSS"}
scored, cheated, lost = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
for r in csv.DictReader(open(gap)):
    k = (r["env"], r["point"], r["who"])
    if r["hacked"] in ("0", "1"): scored[k] += 1; cheated[k] += r["hacked"] == "1"
    else: lost[k][r["status"]] += 1
head = "| environment | point (Llama's run, checkpoint K, takeover) | kind | arm | target | scored | cheated | lost, fixable | lost, unfixable | missing (target - scored - unfixable) |"
sep = "|" + "---|" * 10
out = ["# Hero2 swap loss audit (written by scripts/hero2_swap_foldin.sh)", "",
       "Every restart point of the hero2 swap (results/hero2_swap_plan.json) x arm (llama = Llama 3.3 70B continuing its own run, the control,",
       "fp16 CoreWeave; qwen36 = Qwen3.6 27B, fp8 DeepInfra, thinking off; gemma = gemma-3-27b-it, fp8 DeepInfra, text tools; every arm with",
       "the end-of-run confirmation on): the plan's target (scored replays per point and arm, after top-ups), scored tries, tries that did the",
       "bad act, and tries lost by cause. Fixable = a rerun can replace it (provider failure, crash, budget stop, not graded yet, cut off).",
       "Unfixable = the model's context window, no measure, no value for the measure, step cap without a report. Rows below target are listed",
       "again at the end. 'stopped' tries are scored, not lost. Source: results/hero2_gap_replays.csv.", "", head, sep]
below, tot = [], collections.Counter()
for p in S.PLAN:
    name = S.point_name(p)
    for who in S.WHO:
        k = (p["env"], name, who); target = p.get("target", p.get("reps", 10))
        fx = "; ".join(f"{s}: {c}" for s, c in sorted(lost[k].items()) if s not in UNFIX)
        ux = "; ".join(f"{s}: {c}" for s, c in sorted(lost[k].items()) if s in UNFIX)
        n_ux = sum(c for s, c in lost[k].items() if s in UNFIX)
        miss = max(0, target - scored[k] - n_ux)
        kind = f"cheating run, takeover at {p.get('take', 'N')}" if p["incident"] else "clean run, takeover at its last turn"
        row = f"| {p['env']} | {name} | {kind} | {ARM.get(who, who)} | {target} | {scored[k]} | {cheated[k]} | {fx} | {ux} | {miss or ''} |"
        out.append(row); tot["target"] += target; tot["scored"] += scored[k]; tot["lost"] += sum(lost[k].values()); tot[who] += scored[k]
        if scored[k] < target: below.append(row)
out += ["", f"Total: target {tot['target']}, scored {tot['scored']} (" + ", ".join(f"{ARM.get(w, w)} {tot[w]}" for w in S.WHO) + f"), lost {tot['lost']}.", "",
        f"## Below target: {len(below)} cells", "", head, sep] + below
(OUT / "hero2_swap_loss_audit.md").write_text("\n".join(out) + "\n")
print(f"audit: target {tot['target']}, scored {tot['scored']}, lost {tot['lost']}, {len(below)} cells below target -> {OUT / 'hero2_swap_loss_audit.md'}")
PY
}

# 7. the chart (and its markdown table)
chart () {
  (cd "$ENVS" && $R --with matplotlib python ../scripts/hero2_gap_chart.py) > "$L/chart.txt" 2>&1; local rc=$?
  tail -2 "$L/chart.txt" | while read -r line; do say "chart: $line"; done
  return $rc
}

APPS_DONE=0; REDACTED=0
if [ "$DRY" = 1 ]; then
  say "dry run: not waiting, not fetching, not redacting"
else
  step wait wait_apps && APPS_DONE=1
  if [ $APPS_DONE = 1 ]; then step fetch fetch; else say "fetch skipped: the swap is not launched or our apps have not all stopped"; fi
  step redact redact && REDACTED=1
fi
if [ "$DRY" = 1 ] || [ $REDACTED = 1 ]; then step grade grade; else say "grading skipped: redaction did not succeed"; fi
step score score
step audit audit
step chart chart
say "done (not committed: the parent session commits; outputs: $OUTD/hero2_swap.csv, hero2_swap_validity.csv, hero2_swap_endings.csv, hero2_gap_replays.csv, hero2_swap_loss_audit.md; $CHART)"
[ "$DRY" = 1 ] && echo "dry-run outputs in $SCR"
exit 0
