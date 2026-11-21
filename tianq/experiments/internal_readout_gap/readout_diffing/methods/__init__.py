"""Ways to score how alike two models' readouts are. One file per family of methods. The single place they are written:
the J-space runs (jspace/) and the NLA runs (nla/) both import from here.

  word_overlap.py     every word counts the same                         (methods 1, 2)
  rank_weighted.py    words near the top count more                      (method 3)
  value_weighted.py   a word counts by its probability or its count      (methods 4, 5, 6, 7)
  rare_words.py       words in few lists count more                      (method 8 and its variants)
  meaning.py          synonyms count                                     (method 10)
  pair.py             the input every method takes

Every method is a function of a Pair that returns a table. A table is [n, n]: entry [i, j] is how alike Qwen3.6's
readout at restart point i is to the gemma's readout at restart point j, so the diagonal is "the same restart
point". A table is all evaluation.py needs.

A family's `tables(pair)` returns its methods' tables by label. To add a method: write its function in the fitting
file and list it in a selection below.
"""

from __future__ import annotations

import numpy as np

from . import meaning, rank_weighted, rare_words, value_weighted, word_overlap
from .pair import Pair

SELECTIONS = {
    "all": (word_overlap.tables, value_weighted.dot_products, rare_words.tables, value_weighted.min_and_max,
            rank_weighted.tables, meaning.tables),
    "rare_words": (rare_words.tables,),
    "hybrids": (rare_words.hybrid_tables,),
}


def tables(pair: Pair, selection: str = "all") -> dict[str, np.ndarray]:
    """label -> table, for every method of the chosen selection."""
    out: dict[str, np.ndarray] = {}
    for family in SELECTIONS[selection]:
        out |= family(pair)
    return out
