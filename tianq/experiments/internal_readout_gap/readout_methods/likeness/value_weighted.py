"""Value-weighted: a word counts by its value.

For a J-lens readout the value is the probability the lens gives the word (so: confidence-weighted). For NLA
sentences it is the word's count, and `cosine` is then the cosine of word counts, the first reported NLA score.

  cosine         of the two value vectors
  bhattacharyya  sum over words of sqrt(p_a * p_b)
  min_and_max    probability overlap, sum over words of min(p_a, p_b); and weighted Jaccard, that sum divided by the
                 sum of max(p_a, p_b)
The first two are dot products over every word. The last compares every pair of lists word by word, so it uses each
list's top MIN_MAX_TOP words.
"""

from __future__ import annotations

import numpy as np

from . import cosine as shared_cosine
from .pair import Pair

MIN_MAX_TOP = 1000


def cosine(pair: Pair) -> np.ndarray:
    return shared_cosine.cosine(*pair.matrices())


def bhattacharyya(pair: Pair) -> np.ndarray:
    A, B = pair.matrices()
    return (A.sqrt() @ B.sqrt().T).toarray()


def min_and_max(pair: Pair) -> tuple[np.ndarray, np.ndarray]:
    """(probability overlap, weighted Jaccard)."""
    used = sorted({pair.index[w] for ws, _ in (*pair.a, *pair.b) for w in ws[:MIN_MAX_TOP]})   # the columns any list uses
    Ad, Bd = (m[:, used].toarray() for m in pair.matrices(MIN_MAX_TOP))
    mins = np.stack([np.minimum(Ad[i], Bd).sum(1) for i in range(pair.n)])   # one row of a at a time
    return mins, mins / np.maximum(Ad.sum(1)[:, None] + Bd.sum(1)[None, :] - mins, 1e-12)
