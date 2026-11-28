"""What is read off a likeness table, whichever method made it.

A table is [n, n]: entry [i, j] is how alike row i of one side is to row j of the other (in the hero-swap arena:
Qwen3.6's readout at restart point i and a gemma's at restart point j). The diagonal is "the same row".

  chance              per row, the mean of that row's other entries
  above_chance        same - chance per row: the diagonal minus chance
  same_moment_rates   how often the same row scores above a different one (the test the arena ranks methods by)

"Other entries" is every other row unless a mask narrows it, e.g. to the same task.
A NaN entry means one of the two sides has no readout there (an NLA that wrote nothing): it is left out of chance,
and a row whose own entry is NaN gets NaN.
cluster_ci gives an interval that keeps related rows together when resampling.
"""

from __future__ import annotations

import numpy as np

BOOT, SEED = 2000, 0


def others(n: int) -> np.ndarray:
    """[n, n] mask: every other restart point."""
    return ~np.eye(n, dtype=bool)


def same_task(tasks: list[str]) -> np.ndarray:
    """[n, n] mask: the two restart points are of the same task (a restart point with itself included)."""
    return np.array([[a == b for b in tasks] for a in tasks])


def chance(table: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Per restart point: the mean of its row's other entries. `mask` picks which others (None: every other one)."""
    has = ~np.isnan(table)
    if mask is None:
        return (np.nansum(table, 1) - np.diag(table)) / (has.sum(1) - 1).astype(table.dtype)
    return np.nansum(table * mask, 1) / (has & mask).sum(1)


def above_chance(table: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """same - chance per restart point."""
    return np.diag(table) - chance(table, mask)


def same_moment_rates(table: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Over the masked (restart point, other restart point) pairs: the share where same > other, and the share of ties."""
    same = table.diagonal()[:, None]
    return float((same > table)[mask].mean()), float((same == table)[mask].mean())


def cluster_ci(values: np.ndarray, clusters: list[str]) -> list[float]:
    """95% interval for the mean of `values`, resampling whole clusters with replacement."""
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(np.array(clusters) == c) for c in sorted(set(clusters))]
    means = [values[np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(BOOT)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
