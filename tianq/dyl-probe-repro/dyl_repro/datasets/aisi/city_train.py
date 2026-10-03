"""AISI's own probe-training transcripts, used as a fingerprint, not a benchmark.

The official checkpoint stores `dataset_mean` and `dataset_std`, the
per-dimension statistics of the activations it was trained on (the library
standardises with them, probes.py L129). Replaying these same transcripts, the
right (position, hidden_states index) reproduces those statistics almost
exactly and every wrong one misses by a wide margin. This is how the read
position fixed in core/probe.py was measured, since AISI's extraction code is
not public. The probe's AUROC at each index is reported alongside: this is its
training set, so the right index should score near-perfectly.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from dyl_repro.core.conversation import Conversation
from dyl_repro.core.prefill import POSITIONS
from dyl_repro.core.probe import DylProbe
from dyl_repro.core.store import ActivationCache
from dyl_repro.datasets.aisi import load_split, stored_labels

SPLIT = "dyl_train_city_countries"
SHOW = 10


def conversations() -> list[Conversation]:
    return load_split(SPLIT)


def labels(cache: ActivationCache) -> list[int | None]:
    return stored_labels(cache)


def _statistics(cache: ActivationCache, probe: DylProbe):
    """One pass: per-(position, index) mean and std over the labelled rows, and
    every row's logits at every (position, index)."""
    n, total, total_sq = 0, None, None
    rows, labels, logits = [], [], []
    for record, hidden_states in cache.iter_rows():
        row_logits = probe.score(hidden_states)  # (positions, n_layers + 1)
        rows.append({"uid": record["uid"], "label": record["label"], "logits": row_logits.tolist()})
        if record["label"] is None:
            continue
        x = hidden_states.astype(np.float64)
        total = x if total is None else total + x
        total_sq = x * x if total_sq is None else total_sq + x * x
        n += 1
        labels.append(record["label"])
        logits.append(row_logits)
    if n == 0:
        raise RuntimeError("no labelled rows in the cache")
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean * mean, 0.0))
    return n, mean, std, np.asarray(labels), np.stack(logits), rows


def evaluate(cache: ActivationCache, probe: DylProbe) -> dict:
    mean_ref, std_ref = probe.training_statistics()
    n, mean, std, labels, logits, rows = _statistics(cache, probe)

    table = [
        {
            "position": position,
            "hidden_states_index": k,
            "relative_error": float(np.linalg.norm(mean[p, k] - mean_ref) / np.linalg.norm(mean_ref)),
            "median_z_offset": float(np.median(np.abs(mean[p, k] - mean_ref) / std_ref)),
            "median_std_ratio": float(np.median(std[p, k] / std_ref)),
            "auroc": float(roc_auc_score(labels, logits[:, p, k])),
            "mean_logit": float(logits[:, p, k].mean()),
        }
        for p, position in enumerate(POSITIONS)
        for k in range(mean.shape[1])
    ]
    ranked = sorted(table, key=lambda r: r["relative_error"])

    print(f"\nFINGERPRINT  {SPLIT}  ({n} labelled rows, probe layer {probe.layer})")
    print("mean vs dataset_mean; a match is near 0 relative error, 0 z offset, std ratio 1\n")
    print(f"  {'position':<13} {'hs':>3} {'rel err':>9} {'z offset':>9} {'std ratio':>10} {'auroc':>7} {'mean logit':>11}")
    for r in ranked[:SHOW]:
        print(
            f"  {r['position']:<13} {r['hidden_states_index']:>3} {r['relative_error']:>9.4f} "
            f"{r['median_z_offset']:>9.3f} {r['median_std_ratio']:>10.3f} {r['auroc']:>7.4f} {r['mean_logit']:>11.2f}"
        )
    best = ranked[0]
    print(f"\n  best: {best['position']} at hidden_states[{best['hidden_states_index']}]"
          f"   (core/probe.py reads {probe.hidden_states_index})")
    return {
        "split": SPLIT,
        "n_labelled": n,
        "best": best,
        "table": table,
        "rows_logits_shape": "(positions, n_layers + 1), positions = " + ", ".join(POSITIONS),
        "rows": rows,
    }
