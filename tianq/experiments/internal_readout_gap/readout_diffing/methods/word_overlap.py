"""Word overlap: every word counts the same. Methods 1 and 2.

Jaccard of the two models' top-k word sets: the words in both lists divided by the words in either.
Top 200 is what the first report used.
"""

from __future__ import annotations

import numpy as np

from .pair import Pair

REPORTED_TOP = 200
OTHER_TOPS = (10, 50, 1000, 5000)


def jaccard(pair: Pair, top: int) -> np.ndarray:
    A, B = pair.matrices(top, "one")
    inter = (A @ B.T).toarray()
    union = A.sum(1) + B.sum(1).T - inter
    return np.asarray(inter / np.maximum(union, 1))


def tables(pair: Pair) -> dict[str, np.ndarray]:
    return {f"{'1' if k == REPORTED_TOP else '2'}  Jaccard, top {k:,}": jaccard(pair, k) for k in (REPORTED_TOP, *OTHER_TOPS)}
