"""The input every method takes: two models' readouts at the same restart points, on one shared word index.

A model's readouts are a list with one entry per restart point: (its words, best first; a value for each word).
The value is the lens's probability for a J-lens readout and the word's count for NLA sentences.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

Readouts = list[tuple[list[str], np.ndarray]]


@dataclass
class Pair:
    """`a` is Qwen3.6, `b` a gemma: row i of each is restart point i. `emb` is only needed by the meaning method.
    `pool` is every model's readouts in the experiment (Qwen3.6, base gemma, the final checkpoint): the rare-word
    weights are counted over it, so every comparison of the experiment uses the same weights. Left out, it is a and b."""
    a: Readouts
    b: Readouts
    emb: object | None = None
    pool: list[Readouts] | None = None
    index: dict[str, int] = field(init=False)   # every word in either model's lists -> its column

    def __post_init__(self) -> None:
        self.index = {w: k for k, w in enumerate(sorted({w for ws, _ in (*self.a, *self.b) for w in ws}))}
        if self.pool is None:
            self.pool = [self.a, self.b]

    @property
    def n(self) -> int:
        return len(self.a)

    def matrices(self, top: int | None = None, value: str = "p") -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Both models as [restart points, words] matrices of each list's first `top` words (None: all of them)."""
        return _matrix(self.a, self.index, top, value), _matrix(self.b, self.index, top, value)

    def lists_with_word(self, top: int | None) -> np.ndarray:
        """Per word (column): how many of the pool's lists have it among their first `top` words.
        A third model's words that neither a nor b has are not in the index and are not counted."""
        total = np.zeros(len(self.index), dtype=np.float32)
        for lists in self.pool:
            if lists is not self.a and lists is not self.b:
                lists = [([w for w in ws[:top] if w in self.index], values) for ws, values in lists]
            total += np.asarray(_matrix(lists, self.index, top, "one").sum(0)).ravel()
        return total


def _matrix(lists: Readouts, index: dict[str, int], top: int | None, value: str) -> sparse.csr_matrix:
    """What a word's entry holds: "p" its value, "one" 1, "sqrt p", "rank discount" 1 / log2(rank + 1), else its rank."""
    rows, cols, vals = [], [], []
    for i, (ws, ps) in enumerate(lists):
        k = len(ws) if top is None else min(top, len(ws))
        rows += [i] * k
        cols += [index[w] for w in ws[:k]]
        vals += (ps[:k].tolist() if value == "p" else [1.0] * k if value == "one" else np.sqrt(ps[:k]).tolist() if value == "sqrt p"
                 else (1 / np.log2(np.arange(2, k + 2))).tolist() if value == "rank discount" else list(range(1, k + 1)))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(lists), len(index)), dtype=np.float32)
