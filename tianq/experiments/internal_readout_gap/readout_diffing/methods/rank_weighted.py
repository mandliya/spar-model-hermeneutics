"""Rank-weighted: agreement near the top of the lists counts more. Method 3.

Rank-biased overlap over the top TOP words with persistence P: a word in both lists adds weight(the worse of its two
ranks), where weight(m) = sum over k >= m of P^(k-1) / k, scaled so two identical lists score 1.
"""

from __future__ import annotations

import numpy as np

from .pair import Pair, Readouts

P, TOP = 0.9, 200


def rank_biased_overlap(pair: Pair) -> np.ndarray:
    used = sorted({pair.index[w] for ws, _ in (*pair.a, *pair.b) for w in ws[:TOP]})
    col = {c: k for k, c in enumerate(used)}

    def ranks(lists: Readouts) -> np.ndarray:
        r = np.full((pair.n, len(used)), TOP + 1, dtype=np.int16)   # TOP + 1: the word is not in the list
        for i, (ws, _) in enumerate(lists):
            for rank, w in enumerate(ws[:TOP], 1):
                r[i, col[pair.index[w]]] = rank
        return r

    Ra, Rb = ranks(pair.a), ranks(pair.b)
    terms = np.array([P ** (k - 1) / k for k in range(1, TOP + 1)])
    weight = np.concatenate([[0.0], terms[::-1].cumsum()[::-1], [0.0]]) / (P ** np.arange(TOP)).sum()
    return np.stack([weight[np.maximum(Ra[i], Rb)].sum(1) for i in range(pair.n)])


def tables(pair: Pair) -> dict[str, np.ndarray]:
    return {f"3  rank-biased overlap, top {TOP}, p = {P}": rank_biased_overlap(pair)}
