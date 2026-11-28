"""Meaning: synonyms count. Meaning vectors come from shared.embedding.

  centroid_cosine     for word lists: each list becomes one vector, the probability-weighted mean of its top TOP
                      words' vectors, and two lists are compared by cosine. Needs `pair.emb`.
  sentence_centroids  for NLA sentences: each row becomes the mean of its sentences' vectors
  centred_cosine      two sets of row vectors compared by cosine, after an optional shift (e.g. the average
                      sentence of every model, so that what all readouts share does not count)
"""

from __future__ import annotations

import numpy as np

from ..readout_type import Readouts
from .pair import Pair

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


def sentence_centroids(emb, rows: list[list[str]]) -> np.ndarray:
    """[rows, dimensions]: per row, the mean of its sentences' meaning vectors (`emb` must already hold them)."""
    return np.stack([np.mean(emb.matrix(sentences), axis=0) for sentences in rows])


def centred_cosine(a: np.ndarray, b: np.ndarray, shift: np.ndarray | float = 0.0) -> np.ndarray:
    """The cosine table of two sets of row vectors, each with `shift` taken off first."""
    unit = lambda v: (v - shift) / np.linalg.norm(v - shift, axis=1, keepdims=True)
    return unit(a) @ unit(b).T
