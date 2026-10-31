"""CPU: the transfer matrix with no follow-up question (experiments/cross_category_transfer/no_followup.py)

    python -m experiments.cross_category_transfer.compare_no_followup --plan   # the train_probe runs behind it, one per line
    python -m experiments.cross_category_transfer.compare_no_followup          # the three matrices, from the stored results

--plan is what run_no_followup.sh runs train_probe on. Without it, every training
and test result is read from the Hub (no activations, nothing retrained), and the
three matrices (one per read position) are printed and stored as
gigi/results/no_followup_<recipe>.json, drawn as gigi/charts/no_followup_<recipe>.png.
A missing training result stops it; a missing test result shows as "-".
"""

from __future__ import annotations

import argparse
import os

from dyl_repro.core.store import SCRATCH, RowStore
from experiments.cross_category_transfer import generalization, generalization_charts, no_followup
from experiments.cross_category_transfer.compare_generalization import RESULT, fetch

POSITION_TITLES = {
    "prompt_last": "Just before the reply: is Gemma about to lie?",
    "reply_last": "The reply's last token: the finished reply",
    "reply_mean": "Mean over the reply's tokens",
}
HEADLINE = "No \"did you lie?\": does a probe on the reply itself catch lies in other kinds of deception?"
CAPTION = ("AUROC of lies against truthful replies (Gigi's label for the original reply), with no follow-up question"
           " asked. AISI's probe recipe at layer 58, read at each position. Outlined: tested on its own category"
           " (5-fold cross-validated). Small numbers: lies/truthful. Faded: fewer than"
           f" {generalization.FEW} of either. Gray 0.5 = chance, blue = catches them, red = ranks them backwards.")
NAME = RESULT.replace("generalization_", "no_followup_", 1)


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", action="store_true", help="print the train_probe arguments, one run per line")
    if parser.parse_args().plan:
        for train, tests, position in no_followup.plan():
            print(f"--train {train} --position {position} --test {' '.join(tests)}")
        return

    store = RowStore(os.environ["HF_DATASET_REPO"], os.environ["HF_TOKEN"], SCRATCH, "gigi")
    stored = store.stored()
    matrices = []
    for position in no_followup.POSITIONS:
        trained, tested = {}, {}
        for train in no_followup.CATEGORIES:
            training_path, test_paths = no_followup.result_paths(position, train)
            if training_path not in stored:
                raise SystemExit(f"no training result for {train} at {position}: {training_path}")
            trained[train] = fetch(training_path)
            tested[train] = {test: fetch(path) if path in stored else None for test, path in test_paths.items()}
        matrices.append(no_followup.matrix(position, trained, tested))

    summaries = {m["position"]: generalization.summary(m, (no_followup.MEASURE,)) for m in matrices}
    tables = {m["position"]: generalization.markdown(m, no_followup.MEASURE) for m in matrices}
    for position, table in tables.items():
        s = summaries[position]
        d = s[no_followup.MEASURE]
        print(f"\nNO FOLLOW-UP, read at {position}: {s['probes_trained']} probes trained; over cells with at least"
              f" {generalization.FEW} of each class: in-category {_number(d['mean_in_category'])},"
              f" transfer {_number(d['mean_transfer'])} ({d['transfer_cells_at_least_0_7']} of {d['transfer_cells']}"
              " cells at 0.7 or more)\n")
        print(table)

    panels = [(POSITION_TITLES[m["position"]], m) for m in matrices]
    store.put_chart(NAME, generalization_charts.draw(panels, no_followup.MEASURE, HEADLINE, CAPTION))
    store.put_result(NAME, {"matrices": matrices, "summary": summaries, "markdown": tables})


if __name__ == "__main__":
    main()
