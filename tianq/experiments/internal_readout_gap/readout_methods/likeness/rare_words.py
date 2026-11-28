"""Rare words count more. The best method on the hero-swap test (at the top 5,000 words).

Each list becomes a vector: a word's worth (word_weights/value.py) times its rarity weight (word_weights/rarity.py),
and two lists are compared by cosine. This one function is used for J-lens readouts and for NLA sentences.

The lists the rarity weights are counted over are the pair's pool (pair.py). In the arena every run passes all three
models' readouts (Qwen3.6, base gemma, the final checkpoint) at every restart point, so base gemma and the final
checkpoint are scored with the same weights. (Until 20 Nov the J-space runs counted over the two models being
compared only.)
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..word_weights import rarity
from .cosine import cosine
from .pair import Pair


def rarity_cosine(pair: Pair, top: int | None, value: str = "one") -> np.ndarray:
    """`top`: how far down each list to go (None: all of it). `value`: what a word is worth before its rarity
    weight ("one": 1 if it is in the list; "p": its probability or count; the others: word_weights/value.py)."""
    weights = sparse.diags(rarity.weights(pair.lists_with_word(top), len(pair.pool) * pair.n).astype(np.float32))
    return cosine(*(m @ weights for m in pair.matrices(top, value)))
