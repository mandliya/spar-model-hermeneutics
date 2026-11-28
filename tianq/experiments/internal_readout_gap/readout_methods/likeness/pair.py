"""The input every likeness method takes: two sets of readouts, row by row, on one shared word index.

In the hero-swap arena `a` is Qwen3.6 and `b` a gemma, and row i of each is restart point i. A monitor uses the same
thing with `a` the readouts to score and `b` the labelled examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from ..readout_type import Readouts
from ..word_weights import value as word_value


@dataclass
class Pair:
    """`emb` is only needed by the meaning method. `pool` is every set of readouts in the experiment (in the arena:
    Qwen3.6, base gemma, the final checkpoint): the rare-word weights are counted over it, so every comparison of the
    experiment uses the same weights. Left out, it is a and b."""
    a: Readouts
    b: Readouts
    emb: object | None = None
    pool: list[Readouts] | None = None
    index: dict[str, int] = field(init=False)   # every word in either side's lists -> its column

    def __post_init__(self) -> None:
        self.index = {w: k for k, w in enumerate(sorted({w for ws, _ in (*self.a, *self.b) for w in ws}))}
        if self.pool is None:
            self.pool = [self.a, self.b]

    @property
    def n(self) -> int:
        return len(self.a)

    def matrices(self, top: int | None = None, value: str = "p") -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Both sides as [rows, words] matrices of each list's first `top` words (None: all of them). `value`: what
        a word's entry holds ("p", "one", "sqrt p" or "rank discount": word_weights/value.py)."""
        return matrix(self.a, self.index, top, value), matrix(self.b, self.index, top, value)

    def lists_with_word(self, top: int | None) -> np.ndarray:
        """Per word (column): how many of the pool's lists have it among their first `top` words.
        A third set's words that neither a nor b has are not in the index and are not counted."""
        total = np.zeros(len(self.index), dtype=np.float32)
        for lists in self.pool:
            if lists is not self.a and lists is not self.b:
                lists = known(lists, self.index, top)
            total += lists_with_word(lists, self.index, top)
        return total


def matrix(lists: Readouts, index: dict[str, int], top: int | None, value: str) -> sparse.csr_matrix:
    """Readouts as a [rows, words] matrix on a word index: each list's first `top` words (None: all of them), a
    word's entry being its worth (word_weights/value.py). Every word must be in the index (see `known`)."""
    rows, cols, vals = [], [], []
    for i, (ws, ps) in enumerate(lists):
        k = len(ws) if top is None else min(top, len(ws))
        rows += [i] * k
        cols += [index[w] for w in ws[:k]]
        vals += word_value.worth(ps[:k], value)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(lists), len(index)), dtype=np.float32)


def known(lists: Readouts, index: dict[str, int], top: int | None = None) -> Readouts:
    """Each list's first `top` words without the words the index does not have, for readouts the index was not
    built from (a third model's, or the rows a trained monitor scores)."""
    out = []
    for ws, values in lists:
        keep = [k for k, w in enumerate(ws[:top]) if w in index]
        out.append(([ws[k] for k in keep], values[keep]))
    return out


def lists_with_word(lists: Readouts, index: dict[str, int], top: int | None = None) -> np.ndarray:
    """Per word of the index: how many of the lists have it among their first `top` words. This is the count the
    rare-word weights are made from (word_weights/rarity.py), wherever they are used."""
    return np.asarray(matrix(lists, index, top, "one").sum(0)).ravel()
