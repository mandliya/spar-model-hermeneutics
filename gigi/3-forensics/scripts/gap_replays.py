"""One row per swap replay (Qwen continues, or gemma takes over), for the gap analysis in RESULTS.md, 27 Oct 2026.
Joins swap_all_score.py's per-replay verdict (hacked / not), the grader's own verdict where there is one
(honest_success / honest_failure / cheated), the ending (results/swap_endings.csv, with the Haiku labels applied)
and the number of turns the replay ran. Lost replays are kept with their reason, never counted as behaviour.
  python3 gigi/3-forensics/scripts/gap_replays.py   -> results/gap_replays.csv
Hero mode (4 Nov 2026): `gap_replays.py --hero` (or SWAP_HERO=1) reads the hero swap through swap_all_score.py's hero
mode (arms llama / qwen36 / gemma, + gemma4 / gptoss when present), its endings from results/hero_swap_endings.csv
(swap_endings.py --hero; blank until that exists) and writes results/hero_gap_replays.csv. Without --hero nothing changes.
HERO2 mode (4 Nov 2026): `gap_replays.py --hero2` (or SWAP_HERO2=1) does the same for the hero2 swap (swap_all_score.py's
HERO2 mode: results/raw/hero2_swap/, results/hero2_swap_plan.json), endings from results/hero2_swap_endings.csv
(swap_endings.py --hero2; blank until that exists) and writes results/hero2_gap_replays.csv, the same columns as
gap_replays.csv (hero_swap.sh HERO2=1 --topup reads it). Before the swap is launched it says so and exits 0.
"""
import csv, json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import full_runs_table as T
import swap_all_score as S

if S.not_ready():
    print(S.not_ready()); sys.exit(0)
_ends = S.RES / ("hero2_swap_endings.csv" if S.HERO2 else "hero_swap_endings.csv" if S.HERO else "swap_endings.csv")
endings = {r["run"]: r for r in csv.DictReader(open(_ends))} if (_ends.exists() or not S.HERO) else {}
rows = []
for p in S.PLAN:
    for who in S.WHO:
        for fleet in S.fleets(p, who):
            for run in sorted(fleet.glob("run-*")):
                rel = str(run.relative_to(S.ROOT)); st = S.status(run)
                v = verdict = None
                if st == "finished":
                    if p["env"] in T.MEASURES:
                        v = T.MEASURES[p["env"]][1](T.last_state(run), run)
                    elif p["env"] in T.JUDGED:
                        v = T.judged(None, run)
                        g = run / "final" / "grading.json"
                        verdict = json.loads(g.read_text()).get("verdict") if g.exists() else None
                    if rel in S.HAND:   # a verdict read by hand (results/hand_verdicts.csv) replaces the grader's
                        v = S.HAND[rel]; verdict = f"{'cheated' if v else 'honest'} (hand; grader: {verdict})"
                    if v is None:
                        st = S.unscored_reason(run, p["env"])
                e = endings.get(rel, {})
                rows.append({"env": p["env"], "point": S.point_name(p), "incident": p["incident"], "N": p["N"], "K": p["K"],
                             "who": who, "status": st, "hacked": "" if v is None else int(bool(v)),
                             "grader_verdict": verdict or "", "ending": e.get("ending", ""), "turns": e.get("turns", ""),
                             "run": rel})
out = S.RES / ("hero2_gap_replays.csv" if S.HERO2 else "hero_gap_replays.csv" if S.HERO else "gap_replays.csv")
if not rows:
    print(f"no replays found under {S.RAW.relative_to(S.ROOT)}/ for the plan's {len(S.PLAN)} points: {out.name} not written"); sys.exit(0)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} replays -> {out.relative_to(S.ROOT)}")
