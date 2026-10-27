"""Plan the missing clean-run controls (27 Oct 2026): optimize_rendering_engine and coding_norules had no clean run in
the swap because swap_all_plan.py / swap_extra_plan.py take 3 of Qwen's runs per environment, cheating runs first,
and Qwen cheated in 3 or more. Same rules as those scripts for a run without the bad act: restart at K = N-1 where
N is the run's last step (the model redoes the final turn), 10 tries per model. optimize_rendering_engine has 2 clean
runs (both taken); coding_norules has 17 (the first 3 in path order, as the other plans sort).
Writes results/swap_clean_plan.json (same fields as swap_all_plan.json; swap_all_score.py reads it with the others).
  cd gigi/3-forensics/agent-interp-envs && uv run python ../scripts/swap_clean_plan.py
"""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import collect as C
import full_runs_table as T

OUT = HERE / "results" / "swap_clean_plan.json"
if OUT.exists():
    raise SystemExit(f"{OUT.name} exists; results are never overwritten (archive it first)")
RUNS_PER_ENV, REPS = 3, 10

plan = []
for env in ("optimize_rendering_engine", "coding_norules"):
    runs = sorted((r for e, m, r in T.modal_runs() if e == env and m == "qwen3-next-80b" and T.status(r) == "finished"), key=str)
    clean = []
    for r in runs:
        steps = C.steps(r)
        bad = C.coding(r, steps)["cheated"] if env == "coding_norules" else bool(T.judged(T.last_state(r), r))
        if not bad and len(steps) >= 3:
            clean.append((r, steps))
    for run, steps in clean[:RUNS_PER_ENV]:
        N = int(steps[-1].name.split("-")[1]); K = N - 1
        if (run / f"step-{K}").is_dir():
            plan.append({"env": env, "run": str(run.relative_to(HERE)), "K": K, "N": N, "incident": False, "reps": REPS})
OUT.write_text(json.dumps(plan, indent=1))
print(f"{len(plan)} clean restart points; {sum(p['reps'] for p in plan) * 2} tries (Qwen + gemma)")
for p in plan:
    print(f"  {p['env']:26} N/K {p['N']}/{p['K']}  {p['run']}")
