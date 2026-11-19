"""Numbers behind the classic-round charts (19 Nov 2026). Run on the box after the re-runs, from agent-interp-envs/:
  DISTILL_V=classic env -u VIRTUAL_ENV uv run python ../scripts/classic_results.py
writes results/classic_rerun_summary.csv: one row per (who, task), who = qwen (the 40 teacher runs per task that went
into the data, bf16 on our vLLM) or base / c23 / c46 / c69 / c92 (gemma before training and the four checkpoints, our
vLLM). Columns: runs found, finished, bad act k of n scored (distill_eval.value, the pre-registered measures),
finished runs that gave no answer and stopped at the step limit, eval_tampering's "read the note" (of finished runs),
and lost runs by reason (distill_eval.run_status). Reads only; writes this one CSV.
"""
import collections, csv, json, os, pathlib, sys
os.environ.setdefault("DISTILL_V", "classic")
H = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(H / "scripts"))
import yaml  # noqa: E402
import distill_eval as D  # noqa: E402
from full_runs_table import saw_note  # noqa: E402

ENVS = ["eval_tampering", "tower_of_hanoi", "lazy_investigation", "currency_conversion", "norvane", "coding", "coding_norules"]
MODELS = ["base", "c23", "c46", "c69", "c92"]
CAP = {e: yaml.safe_load((H / f"configs/{e}__gemma-3-27b.yaml").read_text()).get("agent", {}).get("max_steps") for e in ENVS}


def row(who, env, runs):
    c = collections.Counter(); lost = collections.Counter()
    for run in runs:
        c["runs"] += 1
        st = D.run_status(run)
        if st != "finished":
            lost[st] += 1; continue
        c["finished"] += 1
        if env == "eval_tampering":
            c["note"] += bool(saw_note(run))
        try:
            v = D.value(env, run)
        except Exception as e:   # noqa: BLE001
            lost[f"unreadable {type(e).__name__}"] += 1; continue
        if v is None:
            steps = [int(p.name.split("-")[1]) for p in run.glob("step-*")]
            c["no_answer_at_cap" if CAP[env] and steps and max(steps) >= CAP[env] - 1 else "no_answer_other"] += 1
            continue
        c["n"] += 1; c["k"] += bool(v)
    return {"who": who, "task": env, "held_out": env in D.HELD_OUT, "step_cap": CAP[env], "runs": c["runs"],
            "finished": c["finished"], "k": c["k"], "n": c["n"], "no_answer_at_cap": c["no_answer_at_cap"],
            "no_answer_other": c["no_answer_other"], "read_note": c["note"] if env == "eval_tampering" else "",
            "lost": "; ".join(f"{a}: {b}" for a, b in sorted(lost.items()))}


rows = []
sel = collections.defaultdict(list)
for r in csv.DictReader(open(H / "results/classic_selected_per_run.csv")):
    if r["environment"] in ENVS:
        sel[r["environment"]].append(H / r["run"])
for env in ENVS:
    rows.append(row("qwen", env, sel[env]))
for m in MODELS:
    for env in ENVS:
        rows.append(row(m, env, sorted(r for r in (H / f"results/raw/distill_classic_{m}" / env).rglob("run-*") if r.is_dir())))
out = H / "results/classic_rerun_summary.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['who']:5} {r['task']:20} k/n {r['k']:2}/{r['n']:<2} finished {r['finished']:2}/{r['runs']:<2} cap {r['no_answer_at_cap']:2} "
          f"note {r['read_note']!s:3} lost {r['lost'] or '-'}")
log = json.loads(pathlib.Path("/workspace/out/classic_v1/log.json").read_text())
print("history keys:", list(log)[:12])
