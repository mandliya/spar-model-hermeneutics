"""Rank-weighted: agreement near the top of the lists counts more.

Rank-biased overlap (Webber et al. 2010) over the top `top` words with persistence `p`: a word in both lists adds
weight(the worse of its two ranks), where weight(m) = sum over k >= m of p^(k-1) / k, scaled so two identical lists
of `top` words score 1. The same number as averaging, over every depth k, the share of the two top-k sets that
agree, with weight p^(k-1).
"""

from __future__ import annotations

import numpy as np

from ..readout_type import Readouts
from .pair import Pair

P, TOP = 0.9, 200


def rank_biased_overlap(pair: Pair, p: float = P, top: int = TOP) -> np.ndarray:
    used = sorted({pair.index[w] for ws, _ in (*pair.a, *pair.b) for w in ws[:top]})
    col = {c: k for k, c in enumerate(used)}

    def ranks(lists: Readouts) -> np.ndarray:
        r = np.full((len(lists), len(used)), top + 1, dtype=np.int16)   # top + 1: the word is not in the list
        for i, (ws, _) in enumerate(lists):
            for rank, w in enumerate(ws[:top], 1):
                r[i, col[pair.index[w]]] = rank
        return r

    Ra, Rb = ranks(pair.a), ranks(pair.b)
    terms = np.array([p ** (k - 1) / k for k in range(1, top + 1)])
    weight = np.concatenate([[0.0], terms[::-1].cumsum()[::-1], [0.0]]) / (p ** np.arange(top)).sum()
    return np.stack([weight[np.maximum(Ra[i], Rb)].sum(1) for i in range(len(pair.a))])
