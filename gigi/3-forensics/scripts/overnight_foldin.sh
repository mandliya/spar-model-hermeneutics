#!/usr/bin/env bash
# Fold in the gap-row top-ups (scripts/topup_gaps.sh, launched 26 Oct ~21:00) unattended, 26-27 Oct 2026.
# Waits until none of the top-up apps is running, then: fetch every fleet, redact the key, grade (Opus 5, then
# Opus 4.8 for refusals; stops before an environment if the key's spend since 18 Oct reaches $1480), score, label
# the new "other" endings, redraw every chart, write the loss audit (results/loss_audit.md: every chart-4 cell,
# scored vs lost, rows under 60 per model flagged), scan for the key, commit and push. Idempotent: rerun it any time
# (e.g. if the laptop slept): every step skips what is already done.
#   caffeinate -i bash gigi/3-forensics/scripts/overnight_foldin.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ENVS="$HERE/agent-interp-envs"; ROOT="$(cd "$HERE/../.." && pwd)"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
L="$HERE/results/logs/overnight_foldin"; mkdir -p "$L"; T="$HERE/results/logs/topup_gaps"
say () { echo "$(date '+%F %T') $*" >> "$L/progress.txt"; }
R="env -u VIRTUAL_ENV uv run"
say "start"
# 1. wait for the launcher, then for every top-up app to stop running (max 14 h)
for i in $(seq 1 840); do grep -q "all launched" "$T/launch.txt" 2>/dev/null && break; sleep 60; done
for i in $(seq 1 840); do
  ids=$(grep -aoh "ap-[A-Za-z0-9]*" "$T"/*.log | sort -u)
  running=$(cd "$ENVS" && $R modal app list --json 2>/dev/null | python3 -c "
import sys, json
ids = set(sys.argv[1].split())
apps = json.load(sys.stdin)
print(sum(1 for a in apps if (a.get('App ID') or a.get('app_id')) in ids and 'stopped' not in str(a.get('State', a.get('state', ''))).lower()))" "$ids")
  [ "${running:-1}" = 0 ] && break
  [ $((i % 30)) = 1 ] && say "waiting: $running top-up apps still running"
  sleep 60
done
say "all top-up apps finished"
# 2. fetch every fleet named in the top-up logs into its results dir
cd "$ENVS" && $R python - "$HERE" <<'PY' >> "$L/progress.txt" 2>&1
import pathlib, re, subprocess, sys
H = pathlib.Path(sys.argv[1]); T = H / "results/logs/topup_gaps"
for line in (T / "commands.txt").read_text().splitlines():
    stream, name, out, step, n, who = line.split("|")
    t = (T / f"{name}.log").read_text(errors="replace"); vol = re.search(r"^Volume: (\S+)", t, re.M)
    for fleet in sorted(set(re.findall(r"(resume/[a-z0-9-]+/[0-9_-]+)/run-", t))):
        dest = H / out / fleet
        if any(dest.glob("run-*/step-*")): continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(["modal", "volume", "get", "--force", vol.group(1), fleet, str(dest.parent)], capture_output=True, text=True)
        print(("fetched " if rc.returncode == 0 else "FETCH FAILED ") + name)
PY
say "fetched"
# 3. redact the key (ai_rd puts it in the agent's shell)
FORENSICS_ENV_FILE="$ROOT/.env" python3 "$HERE/scripts/redact_key.py" "$HERE/results/raw/swap_all" >> "$L/progress.txt" 2>&1
# 4. grade the new rendering and dashboard replays
cd "$HERE" && FORENSICS_ENV_FILE="$ROOT/.env" SWAP_ALL=1 GRADE_CAP_USD=150 TOTAL_CEILING_USD=1480 python3 scripts/grade_modal.py
FORENSICS_ENV_FILE="$ROOT/.env" SWAP_ALL=1 GRADE_CAP_USD=150 TOTAL_CEILING_USD=1480 JUDGE_MODEL=claude-opus-4-8 python3 scripts/grade_modal.py
say "graded"
# 5. score, label new endings, redraw
cd "$ENVS"
$R python ../scripts/swap_all_score.py >> "$L/progress.txt" 2>&1
$R python ../scripts/full_runs_table.py > /dev/null 2>&1
$R python ../scripts/gemma_endings.py > /dev/null 2>&1
$R python ../scripts/swap_validity.py > "$L/swap_validity.txt" 2>&1
$R python ../scripts/swap_endings.py > /dev/null 2>&1
cd "$HERE" && FORENSICS_ENV_FILE="$ROOT/.env" python3 scripts/label_endings_llm.py >> "$L/progress.txt" 2>&1
cd "$ENVS" && $R python ../scripts/swap_endings.py > /dev/null 2>&1
$R --with matplotlib python ../scripts/story_charts.py > "$L/story_charts.txt" 2>&1
say "scored and redrawn"
# 6. loss audit
cd "$HERE" && python3 - <<'PY'
import csv, re, collections, pathlib
rows = list(csv.DictReader(open("results/swap_all.csv")))
out = ["# Loss audit (written by scripts/overnight_foldin.sh)", "",
       "Every swap cell: replays scored vs lost, by cause. Rows of chart 4 (pooled per takeover turn) under 60 scored per model are flagged.", "",
       "| environment | point | incident | model | scored | lost | why lost |", "|---|---|---|---|---|---|---|"]
for r in rows:
    lost = sum(int(x) for x in re.findall(r": (\d+)", r["lost"] or ""))
    out.append(f"| {r['env']} | {r['point'][-14:]} | {r['incident']} | {r['who']} | {r['n']} | {lost} | {r['lost']} |")
gaps = [l for l in open("results/logs/overnight_foldin/story_charts.txt") if " gap " in l]
out += ["", "## Chart 4 rows", "", "```"] + [g.rstrip() for g in gaps] + ["```", ""]
low = [g.rstrip() for g in gaps if any(int(n) < 60 for n in re.findall(r"(?:Qwen|gemma) \d+/(\d+)", g)) and "clean" not in g]
out += ["## Rows under 60 scored per model (excluding pooled clean rows)", ""] + [f"- {g}" for g in low]
pathlib.Path("results/loss_audit.md").write_text("\n".join(out) + "\n")
PY
say "audit written: results/loss_audit.md"
# 7. key scan, commit, push
cd "$ROOT"
K=$(grep '^OPENROUTER_API_KEY=' .env | cut -d= -f2-)
git pull -q
git add gigi/3-forensics/results/*.csv gigi/3-forensics/results/*.jsonl gigi/3-forensics/results/loss_audit.md gigi/3-forensics/charts \
  gigi/3-forensics/results/logs/topup_gaps/launch.txt gigi/3-forensics/results/logs/topup_gaps/commands.txt gigi/3-forensics/results/logs/topup_gaps/watchdog.txt \
  gigi/3-forensics/results/logs/overnight_foldin gigi/3-forensics/results/logs/grading/grade.txt
if git diff --cached | grep -qF -- "$K" || git diff --cached | grep -qE 'sk-or-v1-[0-9a-f]{64}'; then
  say "KEY FOUND IN STAGED FILES: not committing"; git reset -q; exit 1
fi
git check-ignore -q .env || { say ".env not ignored: not committing"; git reset -q; exit 1; }
git commit -q -m "gigi: forensics: gap-row top-ups folded in overnight (fetch, grade, score, charts, loss audit in results/loss_audit.md); RESULTS.md prose to follow

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && say "pushed $(git log --oneline -1 | cut -c1-8)"
say "done"
