"""CPU: score one dataset's stored activations with the official probe.

    python -m dyl_repro.cli.evaluate --dataset aisi_city_train
    python -m dyl_repro.cli.evaluate --dataset aisi_varied_deception
    python -m dyl_repro.cli.evaluate --dataset ascii_sandbagging --followup all

Pulls whatever is missing from the Hub first, runs the dataset's own check, and
stores the result next to the rows. A partial cache is scored but loudly marked,
so a smoke-test number can never pass for a real one.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from dyl_repro import datasets
from dyl_repro.core.probe import DEFAULT_CHECKPOINT, DylProbe
from dyl_repro.core.store import SCRATCH, ActivationCache, RowStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=datasets.NAMES)
    parser.add_argument("--followup", nargs="+", help="for datasets with swappable follow-ups: names, or 'all'")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


@dataclass(frozen=True)
class StoredRun:
    """One dataset run's rows, pulled from the Hub, and whether all of them are there."""

    dataset: str
    key: str
    followup: str | None
    store: RowStore
    cache: ActivationCache
    rows_expected: int

    @property
    def complete(self) -> bool:
        return len(self.cache.records) == self.rows_expected


def pull_run(dataset: str, key: str, followup: str | None) -> StoredRun:
    store = RowStore(os.environ["HF_DATASET_REPO"], os.environ["HF_TOKEN"], SCRATCH, key)
    store.pull()
    run = StoredRun(dataset, key, followup, store, ActivationCache(SCRATCH, key),
                    len(datasets.conversations(dataset, followup)))
    if not run.complete:
        print(f"\n*** INCOMPLETE {key}: {len(run.cache.records)} of {run.rows_expected} rows stored."
              " Not a real result. ***")
    return run


def evaluate_run(run: StoredRun, probe: DylProbe) -> dict:
    """The dataset's own check of `probe`, stored as <checkpoint>_hs<index> next to the rows."""
    result = datasets.evaluate(run.dataset, run.cache, probe, run.followup)
    result |= {"checkpoint": probe.checkpoint, "trained_on": probe.trained_on, "rows_stored": len(run.cache.records),
               "rows_expected": run.rows_expected, "complete": run.complete}
    index = result.get("hidden_states_index")
    name = probe.checkpoint.removesuffix(".pt") + ("" if index is None else f"_hs{index}")
    run.store.put_result(name if run.complete else f"{name}_INCOMPLETE", result)
    return result


def main() -> None:
    args = parse_args()
    probe = DylProbe.load(args.checkpoint)
    results = {
        followup or key: evaluate_run(pull_run(args.dataset, key, followup), probe)
        for key, followup in datasets.runs(args.dataset, args.followup)
    }
    summarize = getattr(datasets.get(args.dataset), "summarize", None)
    if summarize is not None and len(results) > 1:
        summary = summarize(results)
        summary["all_complete"] = all(r["complete"] for r in results.values())
        store = RowStore(os.environ["HF_DATASET_REPO"], os.environ["HF_TOKEN"], SCRATCH, args.dataset)
        store.put_result("summary" if summary["all_complete"] else "summary_INCOMPLETE", summary)


if __name__ == "__main__":
    main()
