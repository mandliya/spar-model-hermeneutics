"""Top-up list for the swap: for every restart point and model, the replays lost to provider errors, crashes or
missing logs (never the ones lost to gemma's context window, the unscored ones awaiting a grader, or environments
without a bad-behaviour measure), as resume commands for scripts/swap_all.sh (results/logs/swap_all/commands_redo.txt).
  python3 gigi/3-forensics/scripts/swap_redo_plan.py
"""
import csv, json, pathlib, shlex
HERE = pathlib.Path(__file__).resolve().parents[1]
plan = {(p["env"], f"{pathlib.Path(p['run']).parent.name}_{pathlib.Path(p['run']).name}_K{p['K']}"): p
        for p in json.load(open(HERE / "results/swap_all_plan.json"))}
NOMEASURE = {"chess", "tictactoe", "puppeteer", "sandbagging"}
GEMMA = "agent.model=google/gemma-3-27b-it,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null,agent.text_tools=true"
lines, total = [], 0
for r in csv.DictReader(open(HERE / "results/swap_all.csv")):
    if r["env"] in NOMEASURE:
        continue
    lost = dict((x.split(": ")[0], int(x.split(": ")[1])) for x in r["lost"].split("; ") if x)
    n_redo = sum(v for k, v in lost.items() if k in ("provider failure", "crash", "no log"))
    if n_redo <= 0:
        continue
    p = plan[(r["env"], r["point"])]; run = HERE / p["run"]; total += n_redo
    out = HERE / "results/raw/swap_all" / r["env"] / r["point"] / r["who"]
    log = HERE / "results/logs/swap_all" / f"{r['env']}__{r['point']}__{r['who']}__redo.log"
    args = ["--step-dir", str(run / f"step-{p['K']}"), "--count", str(n_redo), "--use-current", "--max-concurrent", "0",
            "--no-judge", "--timeout", "7200", "--results-dir", str(out)] + (["--overrides", GEMMA] if r["who"] == "gemma" else [])
    lines.append(shlex.quote(str(log)) + " " + " ".join(shlex.quote(a) for a in args))
(HERE / "results/logs/swap_all/commands_redo.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
print(f"{len(lines)} resumes, {total} replays to top up")
