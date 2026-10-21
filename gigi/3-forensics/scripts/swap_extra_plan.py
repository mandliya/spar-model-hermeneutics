"""Plan the swap for the five environments added to the full runs on 21 Oct (stage full_extra: coding, coding_norules,
norvane, precommit_hook, secret_number), with swap_all_plan.py's rules: up to 3 of Qwen's finished runs per
environment, incidents first; N = the step of the bad act (collect.py's first_cheat_step: the first write of a
solution.py that special-cases 7, the first hook bypass, the first read of the answer file), else the run's last
step; restart at N-1, and N-2 for incidents. norvane's runs are 2 steps (cap 5), so its restart point is checkpoint 0
(after Qwen's first command), and its runs need only 2 steps. precommit_hook's incidents are left out: its swap at the incident ran
in detail on 17 Oct (gemma rerun in text mode on 20 Oct, RESULTS.md), so here it gets Qwen's final decisions only.
Writes results/swap_extra_plan.json (same fields as swap_all_plan.json; swap_all_score.py reads both).
  python3 gigi/3-forensics/scripts/swap_extra_plan.py
"""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import collect as C
import full_runs_table as T

ENVS = ["coding", "coding_norules", "norvane", "precommit_hook", "secret_number"]
RUNS_PER_ENV, REPS = 3, 10
OUT = HERE / "results" / "swap_extra_plan.json"
if OUT.exists():
    raise SystemExit(f"{OUT.name} exists; results are never overwritten (archive it first)")

plan = []
for env in ENVS:
    runs = [r for e, m, r in T.modal_runs() if e == env and m == "qwen3-next-80b" and T.status(r) == "finished"
            and len(C.steps(r)) >= (2 if env == "norvane" else 3)]
    fn = getattr(C, "coding" if env.startswith("coding") else env)
    scored = [(r, fn(r, C.steps(r))) for r in runs]
    if env == "precommit_hook":
        scored = [(r, s) for r, s in scored if not s["cheated"]]
    scored.sort(key=lambda x: (not x[1]["cheated"], str(x[0])))
    for run, s in scored[:RUNS_PER_ENV]:
        steps = C.steps(run); incident = bool(s["cheated"])
        N = s["first_cheat_step"] if incident and s.get("first_cheat_step") is not None else int(steps[-1].name.split("-")[1])
        for K in ([N - 1, N - 2] if incident else [N - 1]):
            if K >= (0 if env == "norvane" else 1) and (run / f"step-{K}").is_dir():
                plan.append({"env": env, "run": str(run.relative_to(HERE)), "K": K, "N": N, "incident": incident, "reps": REPS})
OUT.write_text(json.dumps(plan, indent=1))
print(f"{len(plan)} restart points in {len({p['env'] for p in plan})} environments; {sum(p['reps'] for p in plan) * 2} replays")
for env in ENVS:
    ps = [p for p in plan if p["env"] == env]
    print(f"  {env:16} points={len(ps):2} incidents={sum(p['incident'] for p in ps):2}  N/K: " +
          ", ".join(f"{p['N']}/{p['K']}" for p in ps))
