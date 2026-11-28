"""Loading J-lens readouts: one read's top 5,000 words with their probabilities, at every stored depth and row.

Input files are write_jlens_lists.py's, in the folder READOUT_DISTRIBUTIONS: for each read and lens, the lens's top
5,000 tokens and their log-probabilities at every stored depth and row, and once per model its token table.

Tokens become words as everywhere else (stripped, lower-cased); probabilities of tokens that become the same word are
added. "masked" also drops special tokens and words without a letter or digit (Qwen3.6's list is topped by
<|endoftext|> at most rows up to halfway).

`Read.lists` gives a read's readouts at one depth in the one form every method takes (readout_type.py).
"""

from __future__ import annotations

import json
import os
import pathlib
import string
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # internal_readout_gap/: shared
from shared import readout  # noqa: E402

FOLDER = pathlib.Path(os.environ["READOUT_DISTRIBUTIONS"])


def normal(token: str) -> str:
    return token.strip().lower()


def keep(word: str, masked: bool) -> bool:
    """shared.words.lens_words's rule (not empty, not punctuation only); masked: also no special token, and a letter or digit."""
    if not word or all(c in string.punctuation for c in word):
        return False
    return not masked or (not (word.startswith("<") and word.endswith(">")) and any(c.isalnum() for c in word) and "�" not in word)


class Read:
    """One read through one lens: per depth and row, its words (best first) and their probabilities."""

    def __init__(self, model: str, lens: str, vocabulary: str | None = None):
        """`vocabulary`: the model whose token table the file's ids are in, when `model` names a read and not a model."""
        vocabulary = vocabulary or model
        z = np.load(FOLDER / f"{model}.{lens}.npz")
        self.names = [str(n) for n in z["names"]]
        self.layers = [int(x) for x in z["layers"]]
        self.ids, self.logp, self.mass, self.entropy = z["ids"], z["logp"], z["mass"], z["entropy"]
        self.vocab = [normal(t) for t in json.loads((FOLDER / f"vocab_{vocabulary}.json").read_text())]
        self.n_layers = 62 if "gemma" in vocabulary else 64

    def select(self, names: list[str]) -> None:
        """Keep only these rows, in this order."""
        rows = [self.names.index(n) for n in names]
        self.names, self.ids, self.logp = list(names), self.ids[rows], self.logp[rows]
        self.mass, self.entropy = self.mass[rows], self.entropy[rows]

    def lists(self, depth: float, masked: bool) -> list[tuple[list[str], np.ndarray]]:
        j = self.layers.index(readout.layer_at(self.n_layers, self.layers, depth))
        out = []
        for i in range(len(self.names)):
            merged: dict[str, float] = {}
            for t, lp in zip(self.ids[i, j], self.logp[i, j]):
                w = self.vocab[t]
                if keep(w, masked):
                    merged[w] = merged.get(w, 0.0) + float(np.exp(lp))
            ws = sorted(merged, key=merged.get, reverse=True)
            out.append((ws, np.array([merged[w] for w in ws], dtype=np.float32)))
        return out
