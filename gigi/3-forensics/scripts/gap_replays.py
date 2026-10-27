"""One row per swap replay (Qwen continues, or gemma takes over), for the gap analysis in RESULTS.md, 27 Oct 2026.
Joins swap_all_score.py's per-replay verdict (hacked / not), the grader's own verdict where there is one
(honest_success / honest_failure / cheated), the ending (results/swap_endings.csv, with the Haiku labels applied)
and the number of turns the replay ran. Lost replays are kept with their reason, never counted as behaviour.
  python3 gigi/3-forensics/scripts/gap_replays.py   -> results/gap_replays.csv
"""
import csv, json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import full_runs_table as T
import swap_all_score as S

endings = {r["run"]: r for r in csv.DictReader(open(HERE / "results" / "swap_endings.csv"))}
rows = []
for p in S.PLAN:
    for who in ("qwen", "gemma"):
        for fleet in S.fleets(p, who):
            for run in sorted(fleet.glob("run-*")):
                rel = str(run.relative_to(HERE)); st = S.status(run)
                v = verdict = None
                if st == "finished":
                    if p["env"] in T.MEASURES:
                        v = T.MEASURES[p["env"]][1](T.last_state(run), run)
                    elif p["env"] in T.JUDGED:
                        v = T.judged(None, run)
                        g = run / "final" / "grading.json"
                        verdict = json.loads(g.read_text()).get("verdict") if g.exists() else None
                    if v is None:
                        st = S.unscored_reason(run, p["env"])
                e = endings.get(rel, {})
                rows.append({"env": p["env"], "point": S.point_name(p), "incident": p["incident"], "N": p["N"], "K": p["K"],
                             "who": who, "status": st, "hacked": "" if v is None else int(bool(v)),
                             "grader_verdict": verdict or "", "ending": e.get("ending", ""), "turns": e.get("turns", ""),
                             "run": rel})
out = HERE / "results" / "gap_replays.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} replays -> {out.relative_to(HERE)}")
