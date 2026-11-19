"""Cost and length of every run in some stages, from the per-call usage ledger (phase 2, 19 Nov 2026).

The OpenRouter provider prints one `[provider-usage]` line per model call into the run's rollout.log (every billed call,
including replies that were then retried) and keeps the same record on each stored assistant message (`_usage` in
messages.json; only kept replies). This script sums both per run.

  python3 scripts/run_usage.py p2_ds_pilot[,p2_ds_smoke] [--csv results/p2_usage__deepseek-v4-pro.csv]

Per run: calls, total cost ($), prompt / completion / reasoning / cached tokens, and the run's length: the largest prompt
of any call (the history size the model last saw, i.e. how close the run came to a context window), and the seconds
between its first and last call (a lower bound on wall clock, for TIMEOUT_S). Per task: runs,
median and max cost per run, median and max length. Prints a projection for N runs per task when --project is given.
"""
import argparse
import csv
import json
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
LINE = re.compile(r"^\[provider-usage\] (.*)$", re.M)
FIELD = re.compile(r"(\w+)=(\S+)")


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def run_usage(run: pathlib.Path) -> dict | None:
    log = run / "rollout.log"
    recs = []
    if log.exists():
        for m in LINE.finditer(log.read_text(errors="replace")):
            recs.append({k: v for k, v in FIELD.findall(m.group(1))})
        recs = [{"in": num(r.get("in")), "out": num(r.get("out")), "reasoning": num(r.get("reasoning")),
                 "cache_read": num(r.get("cache_read")), "cost": num(r.get("cost")), "t": num(r.get("t"))} for r in recs]
        source = "rollout.log"
    if not recs:   # no log lines (older harness or log missing): the kept replies from the last step's messages
        steps = sorted(run.glob("step-*/messages.json"), key=lambda p: int(p.parent.name.split("-")[1]))
        if not steps:
            return None
        msgs = json.loads(steps[-1].read_text())
        msgs = msgs if isinstance(msgs, list) else msgs.get("messages", [])
        recs = [{"in": u.get("prompt_tokens"), "out": u.get("completion_tokens"), "reasoning": u.get("reasoning_tokens"),
                 "cache_read": u.get("cached_tokens"), "cost": u.get("cost"), "t": u.get("t")}
                for u in (m.get("_usage") for m in msgs if m.get("role") == "assistant") if u]
        source = "messages.json"
        if not recs:
            return None
    s = lambda k: sum(r[k] or 0 for r in recs)
    return {"run": str(run.relative_to(HERE)), "calls": len(recs), "cost": round(s("cost"), 6),
            "cost_missing_calls": sum(r["cost"] is None for r in recs), "prompt_tokens": int(s("in")),
            "completion_tokens": int(s("out")), "reasoning_tokens": int(s("reasoning")), "cached_tokens": int(s("cache_read")),
            "max_prompt_tokens": int(max(r["in"] or 0 for r in recs)),
            "model_seconds": int(max(ts) - min(ts)) if (ts := [r["t"] for r in recs if r.get("t")]) else None,
            "source": source}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages")
    ap.add_argument("--csv")
    ap.add_argument("--project", type=int, help="also project the cost of this many runs per task (e.g. 46)")
    a = ap.parse_args()
    rows = []
    for st in a.stages.split(","):
        for run in sorted(p for p in (HERE / "results/raw" / st).rglob("run-*") if p.is_dir()):
            u = run_usage(run)
            if u:
                u["stage"], u["task"] = st, run.relative_to(HERE / "results/raw" / st).parts[0]
                rows.append(u)
    if not rows:
        sys.exit("no runs with usage found")
    if a.csv:
        out = HERE / a.csv
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print("wrote", out.relative_to(HERE))
    tasks = sorted({r["task"] for r in rows})
    print(f"{'task':34} {'runs':>4} {'median $':>9} {'max $':>8} {'median len':>10} {'max len':>9} {'max hours':>9}")
    total = proj = 0.0
    for t in tasks:
        rs = [r for r in rows if r["task"] == t]
        c = [r["cost"] for r in rs]; L = [r["max_prompt_tokens"] for r in rs]
        total += sum(c)
        if a.project:
            proj += statistics.mean(c) * a.project
        hrs = max((r["model_seconds"] or 0) for r in rs) / 3600
        print(f"{t:34} {len(rs):4} {statistics.median(c):9.3f} {max(c):8.3f} {statistics.median(L):10,.0f} {max(L):9,} {hrs:9.2f}")
    print(f"total spent in these runs: ${total:,.2f}; calls without a cost: {sum(r['cost_missing_calls'] for r in rows)}")
    if a.project:
        print(f"projection, {a.project} runs per task over {len(tasks)} tasks (mean cost per run): ${proj:,.0f}")


if __name__ == "__main__":
    main()
