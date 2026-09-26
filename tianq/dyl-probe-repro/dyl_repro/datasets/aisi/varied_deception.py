"""AISI's validation transcripts: the benchmark the published numbers come from.

Success is matching sweep.json for the checkpoint (for the default layer-58
logistic regression: AUROC 0.973, balanced accuracy 0.927, TPR 0.98, FPR 0.126).
Scored at both stored positions side by side: if the published numbers
reproduce at the probe's read position only, on data it never trained on, they
confirm that position independently. Ambiguous rows are excluded only from the
metrics, as AISI excludes them; every row's score is saved.
"""

from __future__ import annotations

from dyl_repro.core.conversation import Conversation
from dyl_repro.core.metrics import detection_metrics, labelled, per_row, random_direction_null
from dyl_repro.core.prefill import POSITIONS
from dyl_repro.core.probe import PROBE_POSITION, DylProbe, published_metrics
from dyl_repro.core.store import ActivationCache
from dyl_repro.datasets.aisi import load_split

SPLIT = "dyl_validate_varied_deception"


def conversations() -> list[Conversation]:
    return load_split(SPLIT)


def _score_at(cache: ActivationCache, probe: DylProbe, position: str) -> dict:
    uids, labels, activations = cache.matrix(position, probe.hidden_states_index)
    scores = probe.score(activations)
    mask, y = labelled(labels)
    return {
        "metrics": detection_metrics(y, scores[mask], probe.threshold),
        "mean_logit_honest": float(scores[mask][y == 0].mean()),
        "mean_logit_lie": float(scores[mask][y == 1].mean()),
        "random_direction_null": random_direction_null(probe.standardise(activations[mask]), y),
        "rows": per_row(uids, labels, scores, probe.threshold),
    }


def evaluate(cache: ActivationCache, probe: DylProbe) -> dict:
    published = published_metrics(probe.checkpoint)
    by_position = {position: _score_at(cache, probe, position) for position in POSITIONS}

    counts = by_position[PROBE_POSITION]["metrics"]
    print(f"\nVARIED DECEPTION  {SPLIT}  hidden_states[{probe.hidden_states_index}]  threshold {probe.threshold:.4f}")
    print(f"  {counts['n_lie']} lie / {counts['n_honest']} honest (ambiguous excluded); probe reads {PROBE_POSITION}\n")
    print(f"  {'metric':<18} {'published':>10}" + "".join(f" {p:>14}" for p in POSITIONS))
    for name, value in published.items():
        print(f"  {name:<18} {value:>10.4f}" + "".join(f" {by_position[p]['metrics'][name]:>14.4f}" for p in POSITIONS))
    for label in ("honest", "lie"):
        print(f"  {'mean logit ' + label:<18} {'':>10}"
              + "".join(f" {by_position[p]['mean_logit_' + label]:>14.2f}" for p in POSITIONS))
    print(f"  {'random |AUROC-.5|':<18} {'':>10}"
          + "".join(f" {by_position[p]['random_direction_null']['mean_abs_deviation']:>14.4f}" for p in POSITIONS))

    return {
        "split": SPLIT,
        "hidden_states_index": probe.hidden_states_index,
        "threshold": probe.threshold,
        "pinned_position": PROBE_POSITION,
        "published": published,
        "by_position": by_position,
    }
