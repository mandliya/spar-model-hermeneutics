"""CPU: just asking vs the probe across Gigi's datasets, each with its best follow-up wording.

    python -m dyl_repro.cli.compare_wordings

Reads the result evaluate stored for every follow-up of every Gigi dataset (the
current version on the Hub), builds the two tables (datasets/gigi/wordings.py)
and their chart (datasets/gigi/wording_charts.py), prints the tables and stores
them as gigi/results/wordings_<result>.json, the chart as
gigi/charts/wordings_<result>.png. No model and no activations: the stored
per-row scores are enough.
"""

from __future__ import annotations

import os

from dyl_repro import datasets
from dyl_repro.core.probe import DylProbe
from dyl_repro.core.store import SCRATCH, RowStore
from dyl_repro.datasets.gigi import wording_charts, wordings

GIGI_DATASETS = tuple(n for n in datasets.NAMES if datasets.MODULES[n].startswith("dyl_repro.datasets.gigi."))


def main() -> None:
    probe = DylProbe.load()
    result_name = f"{probe.checkpoint.removesuffix('.pt')}_hs{probe.hidden_states_index}"
    repo, token = os.environ["HF_DATASET_REPO"], os.environ["HF_TOKEN"]
    results = {
        dataset: {followup: RowStore(repo, token, SCRATCH, key).result(result_name)
                  for key, followup in datasets.runs(dataset, ["all"])}
        for dataset in GIGI_DATASETS
    }
    comparison = wordings.compare(results, probe.threshold)

    tables = comparison["markdown"]
    print("\nWHITE-BOX PROBE VS BLACK-BOX JUST ASKING, EACH WITH ITS BEST WORDING, HELD OUT"
          " (Gigi's label for the original answer)\n")
    print(tables["best_wording"])
    print("\nHARD-TO-CATCH LIES: THE WHITE-BOX PROBE ON THE DENIALS, BEST WORDING, HELD OUT"
          " (black-box just asking scores 0.5 here by construction)\n")
    print(tables["hard_to_catch"])

    store = RowStore(repo, token, SCRATCH, "gigi")
    store.put_chart(f"wordings_{result_name}", wording_charts.draw(comparison))
    store.put_result(f"wordings_{result_name}", comparison)


if __name__ == "__main__":
    main()
