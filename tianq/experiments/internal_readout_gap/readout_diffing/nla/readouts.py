"""Loading for the NLA runs: one model's NLA sentences at every restart point, as words with counts.

Input: the saved NLA files in READOUT_OUT (shared/fetch.py), read with distillation_similarity/similarity.py's own
loaders: five sentences per model per restart point, pooled into word counts.

`as_readouts` turns a model's word counts into the input every method in readout_methods/ takes: per restart point, its
words (most used first) and each word's count. `compare` runs one method on two models, with all three models'
readouts as the pool the rare-word weights are counted over. `sentence_readouts` and `table` do the same for part of
a model's sentences, which is how a model is compared with itself.
"""

from __future__ import annotations

import collections
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]   # readout_diffing/
sys.path.insert(0, str(ROOT.parent))                 # internal_readout_gap/: readout_methods, and shared (similarity.py imports it)
sys.path.insert(0, str(ROOT.parent / "distillation_similarity"))
from readout_methods.likeness.pair import Pair  # noqa: E402
from readout_methods.readout_type import Readouts, from_counts  # noqa: E402
from similarity import ORIGINAL, VERSIONS, load, nla_counts  # noqa: E402

BASE, FINAL = VERSIONS[0], VERSIONS[-1]
MODELS = (ORIGINAL, BASE, FINAL)
Counts = dict[str, collections.Counter]   # restart point -> its words' counts


def load_counts() -> tuple[dict[str, dict], dict[str, Counts], list[str]]:
    """(the saved NLA files per model, the word counts per model, the restart points every model has a file for)."""
    nla = {t: load(t, "nla") for t in MODELS}
    names = sorted(set.intersection(*(set(v) for v in nla.values())))
    return nla, {t: nla_counts(nla[t]) for t in nla}, names


def as_readouts(counts: Counts, names: list[str]) -> Readouts:
    return [from_counts(counts[name]) for name in names]


def sentence_readouts(nla: dict[str, dict], model: str, sentences: set[int], names: list[str]) -> Readouts:
    """One model's readouts from only some of its five sentences per restart point."""
    return as_readouts(nla_counts(nla[model], sentences), names)


def table(method, a: Readouts, b: Readouts, pool: list[Readouts], **options) -> np.ndarray:
    """One method's table for readouts a against readouts b. NaN where either has no words at that restart point."""
    pair = Pair(a, b, pool=pool)
    out = method(pair, **options)
    out[np.array([not ws for ws, _ in a]), :] = np.nan
    out[:, np.array([not ws for ws, _ in b])] = np.nan
    return out


def compare(method, counts: dict[str, Counts], a: str, b: str, names: list[str], **options) -> np.ndarray:
    """One method's table for model a against model b, with all three models' readouts as the pool."""
    readouts = {t: as_readouts(counts[t], names) for t in MODELS}
    return table(method, readouts[a], readouts[b], list(readouts.values()), **options)
