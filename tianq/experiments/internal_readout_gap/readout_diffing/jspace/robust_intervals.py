"""Do the headline numbers survive two simplifications the report made? (18 Nov, review check, exploratory)

    READOUT_DISTRIBUTIONS=<folder of write_jlens_lists.py's files> python robust_intervals.py

The report's intervals resample restart points as if they were independent, and its chance level is taken over all
tasks. Both are generous: two restart points cut from the same run share almost all their history (currency
conversion is one conversation, 62 times), and agreeing on the task is easier than agreeing on the moment. For the
final checkpoint's change from base gemma (own lenses), at the depths the report quotes, this prints
  chance over all tasks    the report's definition: the mean against every other restart point
  chance within the task   the mean against the other restart points of the same task only
each with three 95% intervals: resampling restart points (the report's), resampling whole source runs (a run's
restart points stay together), and resampling whole tasks (13 tasks, so wide by construction).
Two methods: the top-200 word overlap (Jaccard) and the best method on the yardstick (rarity-weighted, top 5,000).
Reads the lens distributions only; prints every number it writes; writes <folder>/robust_intervals.json.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # readout_diffing/: method_menu, restart_points
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # internal_readout_gap/: readout_methods, shared
from readout_methods.likeness import evaluation, rare_words, word_overlap  # noqa: E402
from readout_methods.likeness.pair import Pair  # noqa: E402
from readouts import BASE, FINAL, FOLDER, LENSES, QWEN, load_shared  # noqa: E402
from restart_points import tasks  # noqa: E402

DEPTHS = (2 / 3, 0.85, 0.95)
METHODS = {"top-200 word overlap": lambda pair: word_overlap.jaccard(pair, 200),
           "rarity-weighted, top 5,000": lambda pair: rare_words.rarity_cosine(pair, 5000)}


def main() -> None:
    chosen = LENSES["own lens"]
    reads, names = load_shared([(QWEN, "published"), *((v, chosen[v]) for v in (BASE, FINAL))])
    n = len(names)
    task = tasks(names)
    run = [f"{t}|{re.sub(r'_K[0-9]+$', '', x)}" for t, x in zip(task, names)]
    other = evaluation.others(n)
    other_same_task = other & evaluation.same_task(task)
    print(f"{n} restart points, {len(set(run))} source runs, {len(set(task))} tasks; a task's other restart points: "
          f"{int(other_same_task.sum(1).min())} to {int(other_same_task.sum(1).max())}", flush=True)
    result: dict = {"restart_points": n, "runs": len(set(run)), "tasks": len(set(task)), "scores": {}}
    for depth in DEPTHS:
        q = reads[(QWEN, "published")].lists(depth, False)
        gemma = {v: reads[(v, chosen[v])].lists(depth, False) for v in (BASE, FINAL)}
        pairs = {v: Pair(q, gemma[v], pool=[q, gemma[BASE], gemma[FINAL]]) for v in (BASE, FINAL)}
        for method, table in METHODS.items():
            sims = {v: table(pairs[v]) for v in (BASE, FINAL)}
            for chance_name, mask in (("chance over all tasks", other), ("chance within the task", other_same_task)):
                above = {v: evaluation.above_chance(sims[v], mask) for v in (BASE, FINAL)}
                change = above[FINAL] - above[BASE]
                cis = {"restart points": evaluation.cluster_ci(change, names), "source runs": evaluation.cluster_ci(change, run),
                       "tasks": evaluation.cluster_ci(change, task)}
                tag = f"depth {depth:.3f} | {method} | {chance_name}"
                result["scores"][tag] = {"change": float(change.mean()), "base_above_chance": float(above[BASE].mean()), "ci": cis}
                print(f"{tag}: base gemma above chance {above[BASE].mean():+.4f}; final checkpoint's change {change.mean():+.4f}; "
                      + "; ".join(f"resampling {k} [{lo:+.4f}, {hi:+.4f}]" for k, (lo, hi) in cis.items()), flush=True)
    path = FOLDER / "robust_intervals.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
