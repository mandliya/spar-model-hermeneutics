"""A word counts by how far the lens lifts it above a baseline (the J space paper's concept score).

From Gurnee et al., "Verbalizable Representations Form a Global Workspace in Language Models" (Transformer
Circuits, 6 July 2026), appendix A.21, which turns a J-lens readout into one number for one concept:
    "the mean lens log-probability of two eval-awareness words—fictional and simulation—minus the mean over one
    hundred unrelated control words (common concrete nouns like recipe, bicycl, guitar, …), averaged across the
    workspace layers."
Their list of one hundred words and their exact layers are not published. CONTROL_WORDS is ours, written on 26 Nov
2026 before any result: one hundred common concrete nouns with nothing to do with agents, tasks, lying or people.

  score          the paper's number itself, for chosen concept words in one readout
  control_level  a readout's level on the control words: the baseline the paper uses
  usual_level    another baseline: each word's own usual level over a pool of readouts
  lifted         every word of every readout by its log-probability above a baseline (words below it count nothing)
A control or concept word outside a readout's list has a probability below everything in the list, and is given the
list's smallest one. Everything here reads a list's first TOP words.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..readout_type import Readouts

TOP = 5000
CONTROL_WORDS = (
    "recipe", "bicycle", "guitar", "window", "bottle", "garden", "kitchen", "ladder", "pillow", "candle",
    "basket", "blanket", "bridge", "bucket", "button", "cabinet", "camera", "carpet", "ceiling", "chimney",
    "curtain", "cushion", "drawer", "engine", "fence", "feather", "hammer", "helmet", "jacket", "kettle",
    "lantern", "mirror", "napkin", "needle", "oven", "paddle", "pencil", "piano", "pocket", "ribbon",
    "saddle", "sandal", "scarf", "shovel", "sofa", "spoon", "stove", "sweater", "towel", "umbrella",
    "violin", "wallet", "wheel", "zipper", "anchor", "apron", "barrel", "bench", "boot", "bowl",
    "brick", "broom", "cable", "canvas", "carrot", "cherry", "clock", "coin", "cork", "crayon",
    "fork", "glove", "harbor", "island", "lemon", "marble", "meadow", "mountain", "orchard", "pebble",
    "pepper", "pond", "pumpkin", "river", "rope", "sail", "shelf", "soap", "spade", "stone",
    "table", "teapot", "tent", "thread", "tunnel", "valley", "wagon", "walnut", "whistle", "yarn")


def control_level(words: list[str], values: np.ndarray, top: int = TOP) -> float:
    """The mean log-probability of the control words in one readout's first `top` words (absent: the list's smallest)."""
    if not len(values[:top]):
        return 0.0
    logp = dict(zip(words[:top], np.log(values[:top])))
    floor = float(np.log(values[:top].min()))
    return float(np.mean([logp.get(w, floor) for w in CONTROL_WORDS]))


def score(words: list[str], values: np.ndarray, concept: tuple[str, ...]) -> float:
    """The paper's number for one readout: the concept words' mean log-probability minus the control words' level."""
    logp = dict(zip(words[:TOP], np.log(values[:TOP])))
    floor = float(np.log(values[:TOP].min()))
    return float(np.mean([logp.get(w, floor) for w in concept])) - control_level(words, values)


def usual_level(pool: list[Readouts], index: dict[str, int]) -> np.ndarray:
    """Per word of `index`: the log of its mean probability over every readout of the pool (not in a list: 0 there)."""
    total, n = np.zeros(len(index)), 0
    for lists in pool:
        for ws, ps in lists:
            n += 1
            known = [(index[w], p) for w, p in zip(ws[:TOP], ps[:TOP]) if w in index]
            if known:
                cols, values = zip(*known)
                total[list(cols)] += values
    return np.log(np.maximum(total / n, 1e-30))


def lifted(lists: Readouts, index: dict[str, int], baseline) -> sparse.csr_matrix:
    """[readouts, words]: each of a list's first TOP words by its log-probability above `baseline(words, values)`
    (one number for the readout, or one per word); words at or below the baseline are left out."""
    rows, cols, vals = [], [], []
    for i, (ws, ps) in enumerate(lists):
        ws, ps = ws[:TOP], ps[:TOP]
        if not ws:
            continue
        lift = np.log(ps) - baseline(ws, ps)
        keep = lift > 0
        rows += [i] * int(keep.sum())
        cols += [index[w] for w, k in zip(ws, keep) if k]
        vals += lift[keep].tolist()
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(lists), len(index)), dtype=np.float32)
