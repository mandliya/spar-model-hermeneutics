"""Readouts as words: the one place that says how J-lens tokens and NLA sentences become comparable word lists.

Two models have different tokenizers, so J-lens tokens are compared as words: stripped, lowercased, and dropped when
empty or punctuation only. NLA sentences are split into lowercase words the same way for every model. Both
experiments use these functions for overlap, for bag-of-words features, and for intervals.
"""

from __future__ import annotations

import collections
import math
import re
import string

import numpy as np

BOOT, SEED = 2000, 0
WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)


def lens_words(readout: dict) -> list[str]:
    """A J-lens readout's tokens ({"words": [...]}) as normalized words, in rank order, without repeats."""
    out, seen = [], set()
    for w in readout["words"]:
        w = w.strip().lower()
        if w and not all(c in string.punctuation for c in w) and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def text_words(texts: list[str]) -> list[str]:
    """Every word of some NLA explanations, lowercased, in order (repeats kept: counts matter)."""
    return [w.lower() for t in texts for w in WORD.findall(t)]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else float("nan")


def cosine(a: collections.Counter, b: collections.Counter) -> float:
    """Cosine between two word-count vectors."""
    dot = sum(v * b[k] for k, v in a.items() if k in b)
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else float("nan")


def mean_ci(values, boot: int = BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """(mean, 2.5%, 97.5%) of the values, the interval from resampling them with replacement; NaNs are left out."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (boot, len(v)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)
