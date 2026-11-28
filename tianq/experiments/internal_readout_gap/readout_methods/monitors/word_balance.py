"""Monitors with no training at all: chosen words for, chosen words against (e.g. female words minus male words).

  count_balance    how many of the words are in `plus`, minus how many are in `minus`. For a plain list of words:
                   the words of NLA sentences, or of a text.
  concept_balance  the J space paper's concept score (word_weights/concept_lift.py) for `plus`, minus the same for
                   `minus`. For a J-lens readout. Each score subtracts the same control level, so it cancels.
"""

from __future__ import annotations

import numpy as np

from ..word_weights import concept_lift


def count_balance(words: list[str], plus: frozenset[str], minus: frozenset[str]) -> float:
    return float(sum(w in plus for w in words) - sum(w in minus for w in words))


def concept_balance(words: list[str], values: np.ndarray, plus: frozenset[str], minus: frozenset[str]) -> float:
    return concept_lift.score(words, values, tuple(plus)) - concept_lift.score(words, values, tuple(minus))
