"""The numbers sweep.json reports, computed the standard way, plus a null.

Everything is sklearn. Labels: 1 = lie, 0 = honest. A row is flagged when its
score is strictly above the probe's threshold, the library's own rule.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)


def per_row(uids: list[str], labels: list[int | None], scores: np.ndarray, threshold: float) -> list[dict]:
    """Every row's score, saved with the aggregate so any number can be re-derived."""
    return [
        {"uid": u, "label": y, "score": float(s), "flagged": bool(s > threshold)}
        for u, y, s in zip(uids, labels, scores, strict=True)
    ]


def labelled(labels: list[int | None]) -> tuple[np.ndarray, np.ndarray]:
    """(mask over all rows, labels of the labelled ones). None = excluded from metrics."""
    mask = np.array([y is not None for y in labels])
    return mask, np.array([y for y in labels if y is not None])


def detection_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    flagged = scores > threshold
    tn, fp, fn, tp = confusion_matrix(labels, flagged, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, flagged)),
        "mcc": float(matthews_corrcoef(labels, flagged)),
        "tpr": float(tp / (tp + fn)),
        "fpr": float(fp / (fp + tn)),
        "n_lie": int(tp + fn),
        "n_honest": int(tn + fp),
    }


def random_direction_null(
    standardised: np.ndarray, labels: np.ndarray, n_directions: int = 300, seed: int = 0
) -> dict[str, float]:
    """How far from 0.5 AUROC random directions get on the same standardised vectors.

    Standardised with the probe's own statistics, so the null lives in the space
    the probe's weight vector does. A probe that is not clearly beyond this is
    not reading anything a random direction could not.
    """
    rng = np.random.default_rng(seed)
    deviations = np.array(
        [
            abs(roc_auc_score(labels, standardised @ rng.standard_normal(standardised.shape[1])) - 0.5)
            for _ in range(n_directions)
        ]
    )
    return {
        "mean_abs_deviation": float(deviations.mean()),
        "p95_abs_deviation": float(np.quantile(deviations, 0.95)),
        "n_directions": n_directions,
    }
