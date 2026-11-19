"""Complete Qwen runs per task in the classic round's generation stages (distill/PLAN_classic.md section 1; 17 Nov 2026).

A run is complete when its rollout.log shows FINAL RESULTS or our server's "context window full" (distill_data.run_end).
Runs of a (stage, task) pair listed in results/logs/classic/set_aside.tsv are left out (amendment of 17 Nov: the first
fleet overloaded the server and cut runs; those tasks are re-run). The table has one row per task: launched, complete,
lost by reason, and how many of the lost runs were cut by time or by request timeouts (a sign of an overloaded server).
With --shortfall it prints the env:count list a top-up round needs to reach TARGET complete runs per task (plus a margin
of 2), for modal_full.sh.

  uv run python ../scripts/classic_counts.py classic_qwen36,classic_qwen36_r2 [--shortfall] [--target 40]
"""
import argparse
import collections
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
from distill_data import run_end   # noqa: E402

ENVS = sorted(p.name.split("__")[0] for p in (HERE / "configs").glob(f"*__{os.environ.get('COUNTS_TAG', 'qwen3.6-27b-bf16')}.yaml"))
# phase 2 (19 Nov 2026): SET_ASIDE_FILE / COUNTS_TAG override the classic round's set-aside list and task list; unset = as before
SET_ASIDE_FILE = pathlib.Path(os.environ.get("SET_ASIDE_FILE", HERE / "results/logs/classic/set_aside.tsv"))
OK = ("finished", "context window full")
CUT = ("cut off by the time limit", "our server timed out")


def set_aside():
    pairs = set()
    if SET_ASIDE_FILE.exists():
        for line in SET_ASIDE_FILE.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                stage, task = line.split("\t")[:2]
                pairs.add((stage, task))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages")
    ap.add_argument("--target", type=int, default=40)
    ap.add_argument("--shortfall", action="store_true")
    args = ap.parse_args()
    skip = set_aside()
    ends = collections.defaultdict(collections.Counter)
    for s in args.stages.split(","):
        for env in ENVS:
            if (s, env) in skip:
                continue
            for run in (HERE / "results" / "raw" / s / env).rglob("run-*"):   # <stage>/<env>/<env>/<model>/<time>/run-N
                if not run.is_dir():
                    continue
                ends[env][run_end(run)] += 1
    if args.shortfall:
        need = []
        for env in ENVS:
            ok = sum(ends[env][k] for k in OK)
            if ok < args.target:
                need.append(f"{env}:{args.target - ok + 2}")
        print(" ".join(need))
        return
    print(f"stages: {args.stages}; set aside: {len(skip)} (stage, task) pairs from {SET_ASIDE_FILE.name}")
    print(f"{'task':34s} {'runs':>5s} {'complete':>9s} {'cut':>4s}  lost by reason")
    tot = collections.Counter()
    for env in ENVS:
        c = ends[env]; ok = sum(c[k] for k in OK); cut = sum(c[k] for k in CUT)
        lost = {k: v for k, v in c.items() if k not in OK}
        tot["runs"] += sum(c.values()); tot["complete"] += ok; tot["cut"] += cut
        print(f"{env:34s} {sum(c.values()):5d} {ok:9d} {cut:4d}  {lost or ''}  (context window full: {c['context window full']})")
    print(f"{'total (' + str(len(ENVS)) + ' tasks)':34s} {tot['runs']:5d} {tot['complete']:9d} {tot['cut']:4d}")


if __name__ == "__main__":
    main()
