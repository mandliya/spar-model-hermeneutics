"""Rare words count more: the weighting that wins the hero-swap test (readout_diffing/, method 8b).

A word's weight is log(lists / lists that have the word): a word in almost every readout says little about any one
of them and counts for little; a word in one readout out of a thousand counts for a lot. "Lists" are the readouts
the weights are counted over (for a comparison: every model's readouts; for a monitor: its training readouts).
"""

from __future__ import annotations

import numpy as np


def weights(lists_with_word: np.ndarray, lists: int) -> np.ndarray:
    """Per word: log(lists / lists that have it). A word no list has is weighted as if one did."""
    return np.log(lists / np.maximum(lists_with_word, 1))
