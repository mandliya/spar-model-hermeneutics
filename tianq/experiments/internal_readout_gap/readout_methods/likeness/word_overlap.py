"""Word overlap: every word counts the same.

Jaccard of the two sides' top-k word sets: the words in both lists divided by the words in either.
"""

from __future__ import annotations

import numpy as np

from .pair import Pair


def jaccard(pair: Pair, top: int) -> np.ndarray:
    A, B = pair.matrices(top, "one")
    inter = (A @ B.T).toarray()
    union = A.sum(1) + B.sum(1).T - inter
    return np.asarray(inter / np.maximum(union, 1))
