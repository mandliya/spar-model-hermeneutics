"""Do the two judges agree? Each sampled run, already graded by one judge, is re-graded by the other on a copy (the
original verdict stays untouched), with the environment's own grader through OpenRouter.

  A: Opus 5 verdicts -> re-graded by Opus 4.8: every Opus 5 'cheated' (up to 20) plus 'honest' ones up to 20 in all;
  B: Opus 4.8 verdicts -> re-graded by Opus 5, in environments where Opus 5 does not refuse (dashboard_perf,
     aj_data_anon, leaked_fix, stripped_engine, tables_coverage_tests): up to 20 'cheated' and 20 'honest'.
Seed 0. Copies: results/raw/judge_agreement/<direction>/<env>/fNNN/run-1 (rollout.log, final/ without grading.json,
config.yaml). Writes results/judge_agreement.csv and prints agreement per direction (a refusal is recorded, not counted).
  FORENSICS_ENV_FILE=... python3 gigi/3-forensics/scripts/judge_agreement.py
"""
import csv, json, os, pathlib, random, re, shutil, subprocess
HERE = pathlib.Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "raw" / "judge_agreement"
KEY = re.search(r"^OPENROUTER_API_KEY=(.+)$", pathlib.Path(os.environ["FORENSICS_ENV_FILE"]).read_text(), re.M).group(1).strip()
JUDGED = {"aj_data_anon", "dashboard_perf", "grade_your_own", "leaked_fix", "lucky_seed", "lucky_seed_readable_grader",
          "optimize_cache_c", "stripped_engine", "tables_coverage_tests"}
OK5 = {"aj_data_anon", "dashboard_perf", "leaked_fix", "stripped_engine", "tables_coverage_tests"}

graded = []   # (env, run, judge, cheated?)
for g in list((HERE / "results/raw/modal").glob("gigi-forensics-*/*/*/*/run-*/final/grading.json")) + \
         list((HERE / "results/raw/swap_all").glob("*/*/*/resume/*/*/run-*/final/grading.json")):
    d = json.loads(g.read_text()); run = g.parent.parent; cfg = run.parent / "config.yaml"
    env = re.search(r"^environment: (\S+)", cfg.read_text(), re.M) if cfg.exists() else None
    if env and env.group(1) in JUDGED and str(d.get("evidence", "")).strip(" ."):
        graded.append((env.group(1), run, d.get("judge"), d.get("verdict") == "cheated"))
rng = random.Random(0)


def pick(judge, envs, n_cheat, n_total):
    c = [x for x in graded if x[2] == judge and x[0] in envs and x[3]]
    h = [x for x in graded if x[2] == judge and x[0] in envs and not x[3]]
    c = rng.sample(c, min(n_cheat, len(c)))
    return c + rng.sample(h, min(max(n_total - len(c), 0), len(h)))


sample = [("A", x, "claude-opus-4-8") for x in pick("claude-opus-5", JUDGED, 20, 20)] + \
         [("B", x, "claude-opus-5") for x in pick("claude-opus-4-8", OK5, 20, 40)]
rows, groups = [], {}
for i, (direction, (env, run, judge, cheated), new_judge) in enumerate(sample):
    fleet = OUT / direction / env / f"f{i:03d}"; dest = fleet / "run-1"
    if not dest.exists():
        dest.mkdir(parents=True); shutil.copy(run.parent / "config.yaml", fleet / "config.yaml")
        if (run / "rollout.log").exists():
            shutil.copy(run / "rollout.log", dest / "rollout.log")
        shutil.copytree(run / "final", dest / "final", ignore=shutil.ignore_patterns("grading.json"))
    groups.setdefault((env, new_judge), []).append(str(fleet))
    rows.append({"direction": direction, "env": env, "source": str(run.relative_to(HERE)), "copy": str(dest.relative_to(HERE)),
                 "first_judge": judge, "first_cheated": cheated, "second_judge": new_judge})
print(f"A: {sum(r['direction'] == 'A' for r in rows)} runs, B: {sum(r['direction'] == 'B' for r in rows)} runs", flush=True)

for (env, new_judge), dirs in groups.items():
    subprocess.run(["env", "-u", "VIRTUAL_ENV", "uv", "run", "python", f"environments/{env}/grade_rollouts.py", *dirs,
                    "--concurrency", "20", "--model", new_judge], cwd=HERE / "agent-interp-envs",
                   env={**os.environ, "JUDGE_VIA_OPENROUTER": "1", "OPENROUTER_API_KEY": KEY},
                   stdout=open(HERE / "results/logs/grading/judge_agreement.log", "a"), stderr=subprocess.STDOUT)

for r in rows:
    g = HERE / r["copy"] / "final" / "grading.json"
    r["second_cheated"] = (json.loads(g.read_text()).get("verdict") == "cheated") if g.exists() else ""
with open(HERE / "results" / "judge_agreement.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for d in ("A", "B"):
    rs = [r for r in rows if r["direction"] == d and r["second_cheated"] != ""]
    agree = sum(r["first_cheated"] == r["second_cheated"] for r in rs)
    pos = [r for r in rs if r["first_cheated"]]
    print(f"{d} ({rows[[r['direction'] for r in rows].index(d)]['first_judge']} -> {rows[[r['direction'] for r in rows].index(d)]['second_judge']}): "
          f"agree {agree}/{len(rs)}; first judge's 'cheated' confirmed {sum(r['second_cheated'] for r in pos)}/{len(pos)}; "
          f"not graded by the second judge (refusal or error): {sum(1 for r in rows if r['direction'] == d) - len(rs)}")
