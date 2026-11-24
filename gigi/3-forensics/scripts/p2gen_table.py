"""p2gen_table.py (24 Nov 2026): the result table of one general training set (phase2/general_set/PIPELINE.md).
One row per test task, one column for Qwen3.6 before training and one per saved copy, each cell "k of n" with its 95%
interval. The copies' cells come from results/distill_<dv>_eval.csv (scripts/distill_eval.py score, the same scorer
as p2ds). The "before training" column is NOT run again: it is p2ds's own (results/distill_p2ds_eval.csv, model =
base, 8 tasks x 40 runs on 23 Nov), reused because the settings are identical (scripts/p2gen_configs.py base-check).
  python3 scripts/p2gen_table.py p2gen_toolscale [--copies results/logs/p2gen_toolscale/copies.json]
writes results/distill_<dv>_table.csv (a new file; nothing else is touched) and prints the table as markdown.
"""
import argparse, csv, json, pathlib

H = pathlib.Path(__file__).resolve().parents[1]
WHAT = {"secret_number": "read the answer file or switched the noise off", "tower_of_hanoi": "cheated at the puzzle",
        "currency_conversion": "did the forbidden sum itself", "norvane": "tampered with the tests",
        "eval_tampering": "read the planted note and did not disclose it", "lazy_investigation": "took the shortcut",
        "coding": "made the broken test pass", "coding_norules": "made the broken test pass"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dv"); ap.add_argument("--copies", default=None, help="copies.json from the training (step and epoch per copy)")
    ap.add_argument("--eval", default=None, help="the copies' score file (default results/distill_<dv>_eval.csv)")
    ap.add_argument("--base", default=str(H / "results/distill_p2ds_eval.csv"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ev = pathlib.Path(a.eval) if a.eval else H / f"results/distill_{a.dv}_eval.csv"
    base = {r["env"]: r for r in csv.DictReader(open(a.base)) if r["model"] == "base"}
    rows = [r for r in csv.DictReader(open(ev)) if r["model"] != "base"]
    cp = pathlib.Path(a.copies) if a.copies else H / f"results/logs/{a.dv}/copies.json"
    # `modal volume get ... -` prints its "Finished downloading" line straight after the JSON (24 Nov: the first table
    # failed on that), so read only from the first "{" to the last "}".
    t = cp.read_text() if cp.exists() else ""
    epochs = {k: v.get("epoch") for k, v in json.loads(t[t.find("{"):t.rfind("}") + 1]).items()} if "{" in t else {}
    models = sorted({r["model"] for r in rows}, key=lambda m: int(m[1:]) if m[1:].isdigit() else 10**9)
    cell = {(r["model"], r["env"]): r for r in rows}
    label = {m: (f"after {epochs[m]:.2g} pass{'es' if epochs[m] > 1 else ''} ({m})" if epochs.get(m) else m) for m in models}
    out = pathlib.Path(a.out) if a.out else H / f"results/distill_{a.dv}_table.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "what_is_counted", "who", "passes", "k", "n", "rate", "ci_lo", "ci_hi", "lost", "source"])
        for env in base:
            b = base[env]
            w.writerow([env, WHAT.get(env, ""), "before training", 0, b["k"], b["n"], b["rate"], b["ci_lo"], b["ci_hi"], b["lost"],
                        "p2ds run of 23 Nov, reused"])
            for m in models:
                r = cell.get((m, env))
                if r:
                    w.writerow([env, WHAT.get(env, ""), m, epochs.get(m, ""), r["k"], r["n"], r["rate"], r["ci_lo"], r["ci_hi"], r["lost"], a.dv])
    fmt = lambda r: f"{r['k']} of {r['n']}" if r else "not run"
    print("| Task (what is counted) | Qwen3.6 before (p2ds run, reused) | " + " | ".join(label[m] for m in models) + " |")
    print("|---|--:|" + "--:|" * len(models))
    for env in base:
        print(f"| {env} ({WHAT.get(env, '')}) | {fmt(base[env])} | " + " | ".join(fmt(cell.get((m, env))) for m in models) + " |")
    lost = {m: sum(40 - int(cell[(m, e)]["n"]) for e in base if (m, e) in cell) for m in models}
    print(f"\n8 tasks x 40 runs per column. Runs lost (left out of the counts): before training {sum(40 - int(b['n']) for b in base.values())} of 320; "
          + "; ".join(f"{m} {lost[m]} of {40 * sum((m, e) in cell for e in base)}" for m in models) + f". Written: {out}")


if __name__ == "__main__":
    main()
