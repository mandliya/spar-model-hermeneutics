"""Clean-data view (19 Nov 2026 call, John's notes): the same rates with the "damaged" tries removed, i.e. finished tries
that repeated the same call 3 times in a row or misused a tool (typed submit into the shell, or wrote its own made-up
tool result; same tests as classic_health.py / distill_charts.health_row). Tries that used every allowed step stay in
(Qwen does that too). Same rule for every model.
  python3 classic_clean.py classic   on the box: Qwen's 40 teacher runs per task + this week's re-runs
  python3 classic_clean.py v2        on the laptop: last week's (swap) re-runs
writes results/<round>_clean_rates.csv: one row per (who, task) with the bad act k of n for all tries and for clean tries
(eval_tampering: read the note, of finished tries), and how many finished tries were dropped.
"""
import collections, csv, json, os, pathlib, re, sys
ROUND = sys.argv[1] if __name__ == "__main__" else os.environ.get("DISTILL_V", "classic")   # importable for damaged()
os.environ["DISTILL_V"] = "classic" if ROUND == "classic" else "v2"
H = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(H / "scripts"))
import distill_eval as D  # noqa: E402
from full_runs_table import saw_note  # noqa: E402

SHELL_SUBMIT = re.compile(r'Function: execute_command\nArguments: \{"command": "(?:[^"]*\|\s*)?\s*submit\b')
CALL = re.compile(r"Function: (\w+)\nArguments: (.*?)\n={20}", re.S)
FAKE_RESULT = re.compile(r"^\s*```tool_output", re.M)


def damaged(run):
    t = (run / "rollout.log").read_text(errors="replace")
    calls = CALL.findall(t)
    rep = any(calls[i] == calls[i + 1] == calls[i + 2] for i in range(len(calls) - 2))
    fake = False
    steps = sorted(run.glob("step-*/messages.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    if steps:
        msgs = json.loads(steps[-1].read_text()); msgs = msgs if isinstance(msgs, list) else msgs.get("messages", [])
        fake = any(m.get("role") == "assistant" and FAKE_RESULT.search(str(m.get("content") or "")) for m in msgs)
    return rep or bool(SHELL_SUBMIT.search(t)) or fake



def main():
    if ROUND == "classic":
        ENVS = ["eval_tampering", "tower_of_hanoi", "lazy_investigation", "currency_conversion", "norvane", "coding", "coding_norules"]
        groups = {}
        sel = collections.defaultdict(list)
        for r in csv.DictReader(open(H / "results/classic_selected_per_run.csv")):
            if r["environment"] in ENVS:
                sel[r["environment"]].append(H / r["run"])
        for e in ENVS:
            groups["qwen", e] = sel[e]
        models = {m: f"distill_classic_{m}" for m in ["base", "c23", "c46", "c69", "c92"]}
    else:
        ENVS = ["eval_tampering", "tower_of_hanoi", "currency_conversion", "lazy_investigation", "coding", "coding_norules"]
        groups = {}
        models = {m: f"distill_v2_{m}" for m in ["base", "c24", "c48", "c72", "c96"]}
    for m, stage in models.items():
        for e in ENVS:
            groups[m, e] = sorted(r for r in (H / "results/raw" / stage / e).rglob("run-*") if r.is_dir() and (r / "rollout.log").exists())

    rows = []
    for (who, e), runs in groups.items():
        c = collections.Counter()
        for run in runs:
            if D.run_status(run) != "finished":
                continue
            bad = damaged(run); c["finished"] += 1; c["dropped"] += bad
            if e == "eval_tampering":
                v = bool(saw_note(run))
            else:
                try:
                    v = D.value(e, run)
                except Exception:   # noqa: BLE001
                    v = None
            if v is None:
                continue
            c["n_all"] += 1; c["k_all"] += bool(v)
            if not bad:
                c["n_clean"] += 1; c["k_clean"] += bool(v)
        rows.append({"round": ROUND, "who": who, "task": e, "finished": c["finished"], "dropped_damaged": c["dropped"],
                     "k_all": c["k_all"], "n_all": c["n_all"], "k_clean": c["k_clean"], "n_clean": c["n_clean"]})
    out = H / f"results/{ROUND}_clean_rates.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['who']:5} {r['task']:20} all {r['k_all']:2}/{r['n_all']:<3} clean {r['k_clean']:2}/{r['n_clean']:<3} dropped {r['dropped_damaged']}/{r['finished']}")


if __name__ == "__main__":
    main()
