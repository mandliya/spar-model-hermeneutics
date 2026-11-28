"""Best-matching sentence: NLA sentences compared one with one, not pooled into one bag of words.

Each sentence is its own readout (a word is worth 1 if the sentence has it) and so its own rare-word vector, built
the way rare_words.py builds one: pair.py's matrix times word_weights/rarity.py's weights, counted over every
sentence of the pool. Each of one side's sentences is matched with the other side's closest sentence by cosine, and
the matches are averaged.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..readout_type import from_words
from ..word_lists.nla_sentences import Sentences
from ..word_weights import rarity
from .cosine import cosine
from .pair import lists_with_word, matrix


def best_match(a: Sentences, b: Sentences, pool: list[Sentences]) -> np.ndarray:
    """[n, n]: the mean over a's sentences at i of the cosine with the closest of b's sentences at j.
    NaN where either side has no sentence."""
    every = [from_words(s) for model in pool for point in model for s in point]
    index = {w: k for k, w in enumerate(sorted({w for ws, _ in every for w in ws}))}
    weight = sparse.diags(rarity.weights(lists_with_word(every, index), len(every)))
    most = max(len(point) for model in (a, b) for point in model)

    def vectors(model: Sentences) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Every sentence as a row, `most` rows per restart point (missing ones empty), and which rows are real."""
        slots = [from_words(point[k] if k < len(point) else []) for point in model for k in range(most)]
        rows = matrix(slots, index, None, "one") @ weight
        return rows, (np.asarray(rows.multiply(rows).sum(1)).ravel() > 0).reshape(len(model), most)

    (A, real_a), (B, real_b) = vectors(a), vectors(b)
    cos = cosine(A, B).reshape(len(a), most, len(b), most)
    cos[:, :, ~real_b] = -np.inf                       # a missing sentence of b is never the closest
    closest = cos.max(3)                               # [n, most, n]: a's sentence against b's best at j
    closest[~real_a] = np.nan
    with np.errstate(invalid="ignore"):
        table = np.nanmean(closest, axis=1)
    table[~real_a.any(1), :] = np.nan
    table[:, ~real_b.any(1)] = np.nan
    return table
