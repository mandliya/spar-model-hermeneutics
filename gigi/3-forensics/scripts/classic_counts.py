"""Complete Qwen runs per task in the classic round's generation stages (distill/PLAN_classic.md section 1; 17 Nov 2026).

A run is complete when its rollout.log shows FINAL RESULTS or our server's "context window full" (distill_data.run_end).
Prints a table per task (launched / complete / lost by reason) and, with --shortfall, the env:count list a top-up round
needs to reach TARGET complete runs per task (plus a margin of 2), for modal_full.sh.

  uv run python ../scripts/classic_counts.py classic_qwen36,classic_qwen36_t1 [--shortfall] [--target 40]
"""
import argparse
import collections
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
from distill_data import run_end   # noqa: E402

ENVS = sorted(p.name.split("__")[0] for p in (HERE / "configs").glob("*__qwen3.6-27b-bf16.yaml"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages")
    ap.add_argument("--target", type=int, default=40)
    ap.add_argument("--shortfall", action="store_true")
    args = ap.parse_args()
    ends = collections.defaultdict(collections.Counter)
    for s in args.stages.split(","):
        for env in ENVS:
            for run in (HERE / "results" / "raw" / s / env).rglob("run-*"):   # <stage>/<env>/<env>/<model>/<time>/run-N
                if not run.is_dir():
                    continue
                ends[env][run_end(run)] += 1
    if args.shortfall:
        need = []
        for env in ENVS:
            ok = ends[env]["finished"] + ends[env]["context window full"]
            if ok < args.target:
                need.append(f"{env}:{args.target - ok + 2}")
        print(" ".join(need))
        return
    print(f"{'task':34s} {'runs':>5s} {'complete':>9s}  lost")
    for env in ENVS:
        c = ends[env]; ok = c["finished"] + c["context window full"]
        lost = {k: v for k, v in c.items() if k not in ("finished", "context window full")}
        print(f"{env:34s} {sum(c.values()):5d} {ok:9d}  {lost or ''}  (context window full: {c['context window full']})")


if __name__ == "__main__":
    main()
