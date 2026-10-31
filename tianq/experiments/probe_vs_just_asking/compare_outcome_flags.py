"""CPU: what the released probe flags on each kind of answer to "did you lie?" (experiments/probe_vs_just_asking/outcome_flags.py)

    python -m experiments.probe_vs_just_asking.compare_outcome_flags

Reads the released probe's stored result for every Gigi category under Gigi's
neutral and named wordings (dyl_repro/cli/evaluate.py wrote them; nothing is recomputed),
prints one table per wording, plus the same double down vs truthful denial
split for the released probe on AISI's own Varied Deception (its lie_reason tells
confessions from truthful denials) as the in-distribution reference, and stores them as
gigi/results/outcome_flags_<checkpoint>.json, drawn as gigi/charts/outcome_flags_<checkpoint>.png.
"""

from __future__ import annotations

import os

from dyl_repro.core.probe import PROBE_POSITION, DylProbe
from dyl_repro.core.store import SCRATCH, ActivationCache, RowStore
from dyl_repro.datasets.aisi import varied_deception
from experiments.cross_category_transfer import generalization
from experiments.probe_vs_just_asking import outcome_flag_charts, outcome_flags

WORDINGS = ("neutral", "named")


def main() -> None:
    probe = DylProbe.load()
    name = f"{probe.checkpoint.removesuffix('.pt')}_hs{probe.hidden_states_index}"
    repo, token = os.environ["HF_DATASET_REPO"], os.environ["HF_TOKEN"]
    breakdowns = {
        wording: {
            dataset: outcome_flags.breakdown(RowStore(repo, token, SCRATCH, generalization.key(dataset, wording)).result(name))
            for dataset in generalization.CATEGORIES
        }
        for wording in WORDINGS
    }
    tables = {w: outcome_flags.markdown(b, generalization.CATEGORIES) for w, b in breakdowns.items()}
    for wording, table in tables.items():
        print(f"\nTHE RELEASED PROBE ON EACH ANSWER GROUP, follow-up {wording!r} (threshold {probe.threshold:.4f})\n")
        print(table)

    # The reference: the same split on AISI's own benchmark, under AISI's own question.
    varied = RowStore(repo, token, SCRATCH, generalization.VARIED_DECEPTION).result(name)
    outcome_of = varied_deception.outcomes(ActivationCache(SCRATCH, generalization.VARIED_DECEPTION))
    reference = generalization.denials([(outcome_of[r["uid"]], r["score"])
                                        for r in varied["by_position"][PROBE_POSITION]["rows"]])
    print(f"\nTHE RELEASED PROBE ON AISI'S VARIED DECEPTION (AISI's question): double down vs truthful denial AUROC"
          f" {reference['auroc']:.4f} ({reference['n_lie']} double downs / {reference['n_honest']} truthful denials)")

    store = RowStore(repo, token, SCRATCH, "gigi")
    store.put_chart(f"outcome_flags_{name}", outcome_flag_charts.draw(breakdowns))
    store.put_result(f"outcome_flags_{name}", {"breakdowns": breakdowns, "markdown": tables,
                                               "varied_deception_double_down_vs_truthful_denial": reference})


if __name__ == "__main__":
    main()
