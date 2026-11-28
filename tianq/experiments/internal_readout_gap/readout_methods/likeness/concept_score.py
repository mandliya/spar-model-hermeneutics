"""Concept score as a way to compare readouts: a word counts by how far the lens lifts it above a baseline
(word_weights/concept_lift.py, from the J space paper), and two readouts are compared by the cosine of those lifts.

  above_control_cosine   the baseline is the readout's level on the paper's kind of control words
  unusual_here_cosine    the baseline is each word's own usual level over every readout of the pool
"""

from __future__ import annotations

import numpy as np

from ..word_weights import concept_lift
from .cosine import cosine
from .pair import Pair


def above_control_cosine(pair: Pair) -> np.ndarray:
    return cosine(*(concept_lift.lifted(lists, pair.index, concept_lift.control_level) for lists in (pair.a, pair.b)))


def unusual_here_cosine(pair: Pair) -> np.ndarray:
    usual = concept_lift.usual_level(pair.pool, pair.index)
    level = lambda ws, ps: usual[[pair.index[w] for w in ws]]
    return cosine(*(concept_lift.lifted(lists, pair.index, level) for lists in (pair.a, pair.b)))
