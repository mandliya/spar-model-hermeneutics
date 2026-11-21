"""Meaning: synonyms count. Method 10.

Each list becomes one vector, the probability-weighted mean of its top TOP words' meaning vectors (shared.embedding),
and two lists are compared by cosine. Needs `pair.emb`; without it this family gives no table.
"""

from __future__ import annotations

import numpy as np

from .pair import Pair, Readouts

TOP, DIMENSIONS = 200, 384


def centroid_cosine(pair: Pair) -> np.ndarray:
    def centroids(lists: Readouts) -> np.ndarray:
        cs = []
        for ws, ps in lists:
            v = (pair.emb.matrix(ws[:TOP]) * ps[:TOP, None]).sum(0) if ws else np.zeros(DIMENSIONS)
            cs.append(v / max(np.linalg.norm(v), 1e-12))
        return np.stack(cs)

    pair.emb.add({w for ws, _ in (*pair.a, *pair.b) for w in ws[:TOP]})
    return centroids(pair.a) @ centroids(pair.b).T


def tables(pair: Pair) -> dict[str, np.ndarray]:
    return {} if pair.emb is None else {f"10 meaning: probability-weighted centroid, top {TOP}": centroid_cosine(pair)}
