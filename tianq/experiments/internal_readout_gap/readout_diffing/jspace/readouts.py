"""Loading for the J-space runs: one model's J-lens readouts at every restart point, as words with probabilities.

Input files are lens_distributions.py's, in the folder READOUT_DISTRIBUTIONS: for each (model, lens), the lens's top
5,000 tokens and their log-probabilities at every stored depth and restart point.

Tokens become words as everywhere else (stripped, lower-cased); probabilities of tokens that become the same word are
added. "masked" also drops special tokens and words without a letter or digit (Qwen3.6's list is topped by
<|endoftext|> at most restart points up to halfway).

A model's readouts at one depth are a list with one entry per restart point: (words, best first; their probabilities).
That list is what every method in ../methods/ takes.
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
QWEN, BASE, FINAL = "qwen3.6-27b", "gemma-3-27b-it", "gemma-distilled-step96"
# which lens each gemma is read through; Qwen3.6 is always read through its published lens
LENSES = {"borrowed lens": {BASE: "published", FINAL: "published"}, "own lens": {BASE: "control-lens", FINAL: "final-lens"}}


def normal(token: str) -> str:
    return token.strip().lower()


def keep(word: str, masked: bool) -> bool:
    """shared.words.lens_words's rule (not empty, not punctuation only); masked: also no special token, and a letter or digit."""
    if not word or all(c in string.punctuation for c in word):
        return False
    return not masked or (not (word.startswith("<") and word.endswith(">")) and any(c.isalnum() for c in word) and "�" not in word)


class Read:
    """One model through one lens: per depth and restart point, its words (best first) and their probabilities."""

    def __init__(self, model: str, lens: str):
        z = np.load(FOLDER / f"{model}.{lens}.npz")
        self.names = [str(n) for n in z["names"]]
        self.layers = [int(x) for x in z["layers"]]
        self.ids, self.logp, self.mass, self.entropy = z["ids"], z["logp"], z["mass"], z["entropy"]
        self.vocab = [normal(t) for t in json.loads((FOLDER / f"vocab_{model}.json").read_text())]
        self.n_layers = 62 if "gemma" in model else 64

    def select(self, names: list[str]) -> None:
        """Keep only these restart points, in this order."""
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


def load_shared(keys: list[tuple[str, str]]) -> tuple[dict[tuple[str, str], Read], list[str]]:
    """The (model, lens) reads asked for, cut down to the restart points every one of them has, in one order."""
    reads = {key: Read(*key) for key in keys}
    names = sorted(set.intersection(*(set(r.names) for r in reads.values())))
    for key, r in reads.items():
        print(f"read | {key[0]} through its {key[1]} lens: {len(r.names)} restart points, {len(names)} shared by every read", flush=True)
        r.select(names)
    return reads, names
