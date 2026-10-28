#!/usr/bin/env bash
# Fold in the loss reruns (scripts/fix_losses.sh, 27 Oct 2026) unattended: wait until none of its apps is running,
# fetch every fleet, redact the key, grade (Opus 5, then Opus 4.8 for refusals; stops before an environment if the
# key's spend since 18 Oct reaches $1371), score, label the new "other" endings, redraw every chart (3, 4, 5A-5E),
# write the FULL loss audit (every restart point, both models: target, scored, lost by cause) to results/loss_audit.md
# and progress.txt, scan for the key, commit and push. Idempotent: rerun it any time; every step skips what is done.
#   caffeinate -i bash gigi/3-forensics/scripts/fix_losses_foldin.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ENVS="$HERE/agent-interp-envs"; ROOT="$(cd "$HERE/../.." && pwd)"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
T="$HERE/results/logs/fix_losses"; L="$T"; CEIL=${CEIL:-1371}   # the key's spend ceiling for grading, $ since 18 Oct
say () { echo "$(date '+%F %T') $*" >> "$L/progress.txt"; }
R="env -u VIRTUAL_ENV uv run"
say "start"
# 1. wait for the launcher, then for every app to stop running (max 14 h)
for i in $(seq 1 840); do grep -q "all launched" "$T/launch.txt" 2>/dev/null && break; sleep 60; done
for i in $(seq 1 840); do
  ids=$(grep -aoh "ap-[A-Za-z0-9]*" "$T"/*.log | sort -u)
  running=$(cd "$ENVS" && $R modal app list --json 2>/dev/null | python3 -c "
import sys, json
ids = set(sys.argv[1].split())
apps = json.load(sys.stdin)
print(sum(1 for a in apps if (a.get('App ID') or a.get('app_id')) in ids and 'stopped' not in str(a.get('State', a.get('state', ''))).lower()))" "$ids")
  [ "${running:-1}" = 0 ] && break
  [ $((i % 30)) = 1 ] && say "waiting: $running apps still running"
  sleep 60
done
say "all apps finished"
# 2. fetch every fleet named in the launch logs into its results dir
cd "$ENVS" && $R python -u - "$HERE" <<'PY' >> "$L/progress.txt" 2>&1
import pathlib, re, subprocess, sys
H = pathlib.Path(sys.argv[1]); T = H / "results/logs/fix_losses"
for line in (T / "commands.txt").read_text().splitlines():
    stream, name, out, step, n, who = line.split("|")
    t = (T / f"{name}.log").read_text(errors="replace"); vol = re.search(r"^Volume: (\S+)", t, re.M)
    for fleet in sorted(set(re.findall(r"(resume/[a-z0-9-]+/[0-9_-]+)/run-", t))):
        dest = H / out / fleet
        # re-fetch every fleet once all apps have stopped: a fleet copied while it was still running leaves run dirs
        # without their final state (22 cells read "state not fetched yet" in the first audit, 27 Oct 21:26, because
        # a "already present" check skipped them). A stamp marks a fleet fetched after the apps stopped, so a restarted
        # pass skips it. Each download gets 60 min and 3 tries: the laptop sleeping mid-download leaves a hung client
        # (a 10-try fleet with 300-step histories is ~10k files, 30+ min; 15 min was too short for lucky_seed K268).
        # Beware: `modal volume get --force` replaces the run dirs, so local-only files (final/grading.json) are lost;
        # fetch only fleets that still need it, never re-fetch a graded one.
        stamp = dest / ".fetched_after_apps_stopped"
        if not vol or stamp.exists(): continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                rc = subprocess.run(["modal", "volume", "get", "--force", vol.group(1), fleet, str(dest.parent)],
                                    capture_output=True, text=True, timeout=3600).returncode
            except subprocess.TimeoutExpired:
                rc = -1
            if rc == 0: stamp.touch(); break
        print(("fetched " if rc == 0 else "FETCH FAILED ") + name + " " + fleet + (f" (attempt {attempt + 1})" if attempt else ""))
PY
say "fetched"
# 3. redact the key (ai_rd puts it in the agent's shell)
FORENSICS_ENV_FILE="$ROOT/.env" python3 "$HERE/scripts/redact_key.py" "$HERE/results/raw/swap_all" >> "$L/progress.txt" 2>&1
# 4. grade the new judged replays (rendering, dashboard, grade_your_own, lucky_seed..., copying_of_answers)
GRADE_CAP=${GRADE_CAP:-100}   # $ per grading run (both grader calls share the ceiling below)
cd "$HERE" && FORENSICS_ENV_FILE="$ROOT/.env" SWAP_ALL=1 GRADE_CAP_USD=$GRADE_CAP TOTAL_CEILING_USD=$CEIL python3 scripts/grade_modal.py >> "$L/grade.txt" 2>&1
FORENSICS_ENV_FILE="$ROOT/.env" SWAP_ALL=1 GRADE_CAP_USD=$GRADE_CAP TOTAL_CEILING_USD=$CEIL JUDGE_MODEL=claude-opus-4-8 python3 scripts/grade_modal.py >> "$L/grade.txt" 2>&1
say "graded"
# 5. score, label new endings, redraw every chart
cd "$ENVS"
$R python ../scripts/swap_all_score.py >> "$L/progress.txt" 2>&1
$R python ../scripts/full_runs_table.py > /dev/null 2>&1
$R python ../scripts/gemma_endings.py > /dev/null 2>&1
$R python ../scripts/swap_validity.py > "$L/swap_validity.txt" 2>&1
$R python ../scripts/swap_endings.py > /dev/null 2>&1
cd "$HERE" && FORENSICS_ENV_FILE="$ROOT/.env" python3 scripts/label_endings_llm.py >> "$L/progress.txt" 2>&1
cd "$ENVS" && $R python ../scripts/swap_endings.py > /dev/null 2>&1
$R --with matplotlib python ../scripts/story_charts.py > "$L/story_charts.txt" 2>&1
$R python ../scripts/gap_replays.py >> "$L/progress.txt" 2>&1
$R python ../scripts/stopped_reasons.py >> "$L/progress.txt" 2>&1   # why the scored 'stopped' tries stopped (chart footnotes)
$R --with matplotlib python ../scripts/gap_analysis.py > "$L/gap_analysis.txt" 2>&1
$R --with matplotlib python ../scripts/loss_charts.py >> "$L/progress.txt" 2>&1   # 5I loss table by cause, 5J stopped reasons
say "scored and redrawn"
# 6. the full loss audit: every restart point of every plan, both models
cd "$ENVS" && $R python - "$HERE" <<'PY' >> "$L/progress.txt" 2>&1
import csv, collections, pathlib, sys
H = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(H / "scripts"))
import swap_all_score as S
UNFIX = {"context window exceeded", "no measure", "no value for the measure", "step cap, no report"}
scored, lost = collections.Counter(), collections.defaultdict(collections.Counter)
for r in csv.DictReader(open(H / "results/gap_replays.csv")):
    k = (r["env"], r["point"], r["who"])
    if r["hacked"] != "": scored[k] += 1
    else: lost[k][r["status"]] += 1
out = ["# Loss audit (written by scripts/fix_losses_foldin.sh)", "",
       "Every restart point of the swap (all plans), both models: target, scored tries, and tries lost by cause. Target = the plan's",
       "reps (10), or 60 for dashboard_costly_failure's Qwen rows. Unfixable = gemma's context window, no measure, no value for the",
       "measure, step cap without a report. Rows below target are listed again at the end.", "",
       "| environment | point | kind | model | target | scored | lost, fixable | lost, unfixable |", "|---|---|---|---|---|---|---|---|"]
below = []
for p in S.PLAN:
    name = S.point_name(p)
    for who in ("qwen", "gemma"):
        k = (p["env"], name, who)
        target = 60 if (p["env"] == "dashboard_costly_failure" and p["incident"] and who == "qwen") else p.get("reps", 10)
        fx = "; ".join(f"{s}: {c}" for s, c in sorted(lost[k].items()) if s not in UNFIX)
        ux = "; ".join(f"{s}: {c}" for s, c in sorted(lost[k].items()) if s in UNFIX)
        row = f"| {p['env']} | {name[-16:]} | {'cheating run' if p['incident'] else 'clean run'} | {who} | {target} | {scored[k]} | {fx} | {ux} |"
        out.append(row)
        if scored[k] < target: below.append(row)
out += ["", f"## Below target: {len(below)} cells", "", "| environment | point | kind | model | target | scored | lost, fixable | lost, unfixable |", "|---|---|---|---|---|---|---|---|"] + below
(H / "results/loss_audit.md").write_text("\n".join(out) + "\n")
print("\n".join(out[-(len(below) + 4):]))
PY
say "audit written: results/loss_audit.md"
# 7. key scan, commit, push
cd "$ROOT"
K=$(grep '^OPENROUTER_API_KEY=' .env | cut -d= -f2-)
git pull -q
git add gigi/3-forensics/results/*.csv gigi/3-forensics/results/*.jsonl gigi/3-forensics/results/*.json gigi/3-forensics/results/loss_audit.md gigi/3-forensics/charts \
  gigi/3-forensics/scripts gigi/3-forensics/results/logs/fix_losses/launch.txt gigi/3-forensics/results/logs/fix_losses/commands.txt \
  gigi/3-forensics/results/logs/fix_losses/watchdog.txt gigi/3-forensics/results/logs/fix_losses/progress.txt gigi/3-forensics/results/logs/fix_losses/grade.txt \
  gigi/3-forensics/results/logs/fix_losses/story_charts.txt gigi/3-forensics/results/logs/fix_losses/gap_analysis.txt gigi/3-forensics/results/logs/fix_losses/swap_validity.txt
if git diff --cached | grep -qF -- "$K" || git diff --cached | grep -qE 'sk-or-v1-[0-9a-f]{64}'; then
  say "KEY FOUND IN STAGED FILES: not committing"; git reset -q; exit 1
fi
git check-ignore -q .env || { say ".env not ignored: not committing"; git reset -q; exit 1; }
git commit -q -m "gigi: forensics: loss reruns folded in (every fleet re-fetched after the apps finished; full audit in results/loss_audit.md); charts redrawn

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && git push -q && say "pushed $(git log --oneline -1 | cut -c1-8)"
say "done"
