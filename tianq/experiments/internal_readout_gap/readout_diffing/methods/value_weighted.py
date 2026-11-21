"""Value-weighted: a word counts by its value. Methods 4, 5, 6 and 7.

For a J-lens readout the value is the probability the lens gives the word (so: confidence-weighted). For NLA
sentences it is the word's count, and `cosine` is then the cosine of word counts, the reported NLA score.

  7  cosine of the two value vectors
  5  Bhattacharyya overlap: sum over words of sqrt(p_a * p_b)
  4  probability overlap: sum over words of min(p_a, p_b)
  6  weighted Jaccard: sum of min(p_a, p_b) / sum of max(p_a, p_b)
5 and 7 are dot products over every word. 4 and 6 compare every pair of lists word by word, so they use each list's
top MIN_MAX_TOP words.
"""

from __future__ import annotations

import numpy as np

from .pair import Pair

MIN_MAX_TOP = 1000


def cosine(pair: Pair) -> np.ndarray:
    A, B = pair.matrices()
    dot = (A @ B.T).toarray()
    na, nb = np.sqrt(A.multiply(A).sum(1)), np.sqrt(B.multiply(B).sum(1))
    return np.asarray(dot / np.maximum(na @ nb.T, 1e-12))


def bhattacharyya(pair: Pair) -> np.ndarray:
    A, B = pair.matrices()
    return (A.sqrt() @ B.sqrt().T).toarray()


def dot_products(pair: Pair) -> dict[str, np.ndarray]:
    return {"5  Bhattacharyya overlap": bhattacharyya(pair), "7  cosine of probabilities": cosine(pair)}


def min_and_max(pair: Pair) -> dict[str, np.ndarray]:
    used = sorted({pair.index[w] for ws, _ in (*pair.a, *pair.b) for w in ws[:MIN_MAX_TOP]})   # the columns any list uses
    Ad, Bd = (m[:, used].toarray() for m in pair.matrices(MIN_MAX_TOP))
    mins = np.stack([np.minimum(Ad[i], Bd).sum(1) for i in range(pair.n)])   # one Qwen3.6 restart point at a time
    return {"4  probability overlap (sum of min)": mins,
            "6  weighted Jaccard (min / max)": mins / np.maximum(Ad.sum(1)[:, None] + Bd.sum(1)[None, :] - mins, 1e-12)}
