"""general_tbench_plan.py (24 Nov 2026): what general_tbench_launch.sh asks between rounds. Reads only; no API calls.

  plan   <stage dir> --tasks-dir <terminal-bench-2> --target 11 [--tasks a,b,c]
         [--exclude d,e]   (tasks left out, e.g. the ones that resemble our 28 tasks)
         -> one line per group to launch: "<runs still missing> <task> <task> ..." (nothing printed = the stage is full)
  status <stage dir> --ledger <file> --cap <dollars>     -> one heartbeat line; exit 2 when the ledger reached the cap
  smoke-check <stage dir> --pin siliconflow              -> exit 1 unless every run passes the checks below

A stage dir holds one Harbor job folder per round (<stage>_r1, _r2, ...); a run is a trial folder in one of them.
A run is COMPLETE when it ended by itself (a plain reply or the step cap) and Harbor wrote its result.json. Runs cut by
the time limit, a provider failure or the spend cap are kept on disk, never trained on, and replaced in the next round
(the same rule as phase2_fleet.sh's set-aside, per run). A task whose container never started in 2 or more tries and
has no complete run is reported as broken and not launched again.
smoke-check: ended by itself; reasoning on at least 80% of the assistant turns; a cost on every billed reply; every reply served by the
pinned host; at least one command run; the verifier gave a reward.
"""
import argparse
import collections
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from general_tbench_convert import load, read_json   # noqa: E402


def trial_dirs(stage: pathlib.Path):
    for job in sorted(p for p in stage.glob("*") if p.is_dir() and not p.name.startswith("_")):
        for t in sorted(p for p in job.iterdir() if p.is_dir()):
            yield t


def task_of(t: pathlib.Path) -> str:
    cfg = read_json(t / "agent/run_config.json") or {}
    name = cfg.get("task") or (read_json(t / "result.json") or {}).get("task_name") or t.name.rsplit("__", 1)[0]
    return str(name).split("/")[-1]


def survey(stage: pathlib.Path):
    done = collections.Counter(); never_started = collections.Counter(); cut = collections.Counter()
    for t in trial_dirs(stage):
        task = task_of(t)
        if not (t / "agent/run_config.json").exists():
            if (t / "result.json").exists():   # Harbor gave up before the agent ran (image, sandbox, setup)
                never_started[task] += 1
            continue
        end = read_json(t / "agent/final/episode_end.json") or {}
        if end.get("ended_by_itself") and (t / "result.json").exists():
            done[task] += 1
        elif end:
            cut[task] += 1
    return done, never_started, cut


def plan(a):
    names = sorted(p.name for p in a.tasks_dir.iterdir() if (p / "task.toml").exists())
    if a.tasks:
        want = a.tasks.split(","); missing = [t for t in want if t not in names]
        if missing:
            sys.exit(f"unknown tasks: {missing}")
        names = want
    names = [t for t in names if t not in (a.exclude or "").split(",")]
    done, never, _ = survey(a.stage) if a.stage.exists() else (collections.Counter(), collections.Counter(), None)
    groups = collections.defaultdict(list); broken = []
    for t in names:
        if done[t] == 0 and never[t] >= 2:
            broken.append(t); continue
        if a.target - done[t] > 0:
            groups[a.target - done[t]].append(t)
    if broken:
        print(f"# broken (container never started in 2+ tries): {' '.join(broken)}", file=sys.stderr)
    for m in sorted(groups, reverse=True):
        print(m, *groups[m])


def ledger_total(path: pathlib.Path) -> float:
    total = 0.0
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                total += float(json.loads(line).get("cost") or 0.0)
            except ValueError:
                pass
    return total


def status(a):
    done, never, cut = survey(a.stage) if a.stage.exists() else ({}, {}, {})
    spent = ledger_total(a.ledger); now = time.time(); r429 = r5 = 0; live = 0
    for t in trial_dirs(a.stage) if a.stage.exists() else []:
        calls = t / "agent/calls.jsonl"
        if not calls.exists():
            continue
        live += not (t / "agent/final/episode_end.json").exists()
        if now - calls.stat().st_mtime > 3600:
            continue
        for line in calls.read_text().splitlines()[-200:]:
            try:
                c = json.loads(line)
            except ValueError:
                continue
            if not c.get("kept") and now - (c.get("t") or calls.stat().st_mtime) < 300:
                r429 += str(c.get("error", "")).startswith("rate_limit"); r5 += str(c.get("error", "")).startswith("server")
    print(f"complete runs {sum(done.values())} in {len(done)} tasks; cut runs {sum(cut.values())}; containers that never "
          f"started {sum(never.values())}; runs in flight {live}; spent ${spent:.2f} of cap ${a.cap:.2f}; "
          f"last 5 min: {r429} rate-limit replies, {r5} server-error replies")
    sys.exit(2 if spent >= a.cap else 0)


def smoke_check(a):
    bad = []; n = 0
    for t in trial_dirs(a.stage):
        if not (t / "agent/run_config.json").exists():
            bad.append(f"{t.name}: the agent never ran"); continue
        n += 1; r = load(t)
        calls = [json.loads(x) for x in (t / "agent/calls.jsonl").read_text().splitlines()] if (t / "agent/calls.jsonl").exists() else []
        billed = [c for c in calls if c.get("prompt_tokens") is not None]
        hosts = {str(c.get("served_by")) for c in billed}
        checks = {
            "ended by itself": r["own"],
            # DeepSeek gave no reasoning on 4% of the turns of our 28-task runs (1,172 of 26,419), so not "every turn"
            "reasoning on at least 80% of the assistant turns": r["assistant_turns"] > 0 and r["turns_without_reasoning"] <= 0.2 * r["assistant_turns"],
            "a real cost on every billed reply": bool(billed) and all(c.get("cost") is not None and not c.get("cost_estimated") for c in billed),
            f"every reply served by {a.pin}": bool(hosts) and all(a.pin.replace("-", "") in h.lower().replace(" ", "").replace("-", "") for h in hosts),
            "at least one command run": r["tool_calls"] >= 1,
            "the verifier gave a reward": r["reward"] is not None,
            "a raw file for every reply": len(list((t / "agent/raw").glob("call-*.json"))) == len(calls),
        }
        for k, ok in checks.items():
            if not ok:
                bad.append(f"{t.name}: FAILED {k} (ended: {r['ended']}, hosts: {sorted(hosts)})")
        print(f"{t.name}: ended {r['ended']}, {r['steps']} steps, reward {r['reward']}, ${float(r['cost'] or 0):.4f}, "
              f"{r['turns_without_reasoning']} of {r['assistant_turns']} turns without reasoning")
    for b in bad:
        print(b)
    print(f"SMOKE {'FAILED' if bad or not n else 'OK'}: {n} runs checked, {len(bad)} problems")
    sys.exit(1 if bad or not n else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("stage", type=pathlib.Path); p.add_argument("--tasks-dir", type=pathlib.Path, required=True)
    p.add_argument("--target", type=int, required=True); p.add_argument("--tasks"); p.add_argument("--exclude")
    p = sub.add_parser("status"); p.add_argument("stage", type=pathlib.Path); p.add_argument("--ledger", type=pathlib.Path, required=True)
    p.add_argument("--cap", type=float, required=True)
    p = sub.add_parser("smoke-check"); p.add_argument("stage", type=pathlib.Path); p.add_argument("--pin", default="siliconflow")
    a = ap.parse_args()
    {"plan": plan, "status": status, "smoke-check": smoke_check}[a.cmd](a)


if __name__ == "__main__":
    main()
