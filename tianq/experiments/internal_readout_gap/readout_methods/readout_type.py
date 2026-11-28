"""What a readout is: the words read off one activation, best first, with a value for each word.

  J-lens readout   the lens's most likely words; the value is the probability the lens gives the word
  NLA readout      the words of the NLA's sentences; the value is how often the word is used

Everything else in readout_methods/ takes readouts in this one form, so a method written for one kind works for the
other. `Readouts` is a list of them, one per row (a restart point, or a conversation's read position).
"""

from __future__ import annotations

import collections

import numpy as np

Readout = tuple[list[str], np.ndarray]
Readouts = list[Readout]


def from_counts(counts: collections.Counter) -> Readout:
    """A readout from word counts: the words, most used first (ties in alphabetical order), and their counts."""
    words = sorted(counts, key=lambda w: (-counts[w], w))
    return words, np.array([counts[w] for w in words], dtype=np.float32)


def from_words(words: list[str]) -> Readout:
    """A readout from a plain run of words (repeats kept), e.g. every word of an NLA's sentences."""
    return from_counts(collections.Counter(words))
