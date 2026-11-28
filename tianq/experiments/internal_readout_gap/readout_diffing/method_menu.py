"""The arena's menu: which likeness methods are ranked on the hero swap, at which settings, under which label.

The methods themselves are in readout_methods/likeness/ (one file per family); nothing is computed here. Each entry
gives a method its number and label, the ones the result files and the reports use:
   1, 2   word overlap: every word counts the same                       (word_overlap.py)
   3      rank-weighted: words near the top count more                   (rank_weighted.py)
   4-7    value-weighted: a word counts by its probability or its count  (value_weighted.py)
   8      rare words count more, and its variants                        (rare_words.py)
          8   top 200, every word in the list weighted by rarity alone
          8b  the same at the top 50 / 1,000 / 5,000 words
          8c  rarity x the word's probability, top 5,000
          8d  rarity x sqrt(probability), top 5,000        (hybrids: softer confidence weights than 8c)
          8e  rarity x a rank discount 1 / log2(rank + 1), top 5,000 and top 200
  10      meaning: synonyms count                                        (meaning.py)
  11, 12  concept score: a word counts by its lift above a baseline      (concept_score.py)
The NLA-only methods 13 to 17 (sentences kept apart) are labelled where they are run, nla/sentence_methods.py.

A family's function returns its methods' tables by label, and `tables(pair, selection)` runs a selection of
families. To add a method: write its function in readout_methods/likeness/, then list it here.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # internal_readout_gap/: readout_methods
from readout_methods.likeness import concept_score, meaning, rank_weighted, rare_words, value_weighted, word_overlap  # noqa: E402
from readout_methods.likeness.pair import Pair  # noqa: E402
from readout_methods.word_weights import concept_lift  # noqa: E402

REPORTED_OVERLAP_TOP = 200   # what the first report used
OTHER_OVERLAP_TOPS = (10, 50, 1000, 5000)
OTHER_RARITY_TOPS = (50, 1000, 5000)   # added after TQ asked whether the cut at 200 is what makes method 8 strong
BEST = "8b rarity-weighted cosine, top 5,000"


def overlap(pair: Pair) -> dict[str, np.ndarray]:
    return {f"{'1' if k == REPORTED_OVERLAP_TOP else '2'}  Jaccard, top {k:,}": word_overlap.jaccard(pair, k)
            for k in (REPORTED_OVERLAP_TOP, *OTHER_OVERLAP_TOPS)}


def dot_products(pair: Pair) -> dict[str, np.ndarray]:
    return {"5  Bhattacharyya overlap": value_weighted.bhattacharyya(pair), "7  cosine of probabilities": value_weighted.cosine(pair)}


def min_and_max(pair: Pair) -> dict[str, np.ndarray]:
    mins, weighted = value_weighted.min_and_max(pair)
    return {"4  probability overlap (sum of min)": mins, "6  weighted Jaccard (min / max)": weighted}


def rarity(pair: Pair) -> dict[str, np.ndarray]:
    out = {"8  rarity-weighted cosine, top 200": rare_words.rarity_cosine(pair, 200)}
    out |= {f"8b rarity-weighted cosine, top {k:,}": rare_words.rarity_cosine(pair, k) for k in OTHER_RARITY_TOPS}
    out["8c rarity x probability cosine, top 5,000"] = rare_words.rarity_cosine(pair, 5000, "p")
    return out


def rarity_hybrids(pair: Pair) -> dict[str, np.ndarray]:
    """Added 18 Nov after TQ asked for rarity combined with rank or confidence; 8b and 8c are repeated beside them."""
    return {"8d rarity x sqrt(probability), top 5,000": rare_words.rarity_cosine(pair, 5000, "sqrt p"),
            "8e rarity x rank discount 1/log2(rank + 1), top 5,000": rare_words.rarity_cosine(pair, 5000, "rank discount"),
            "8e rarity x rank discount 1/log2(rank + 1), top 200": rare_words.rarity_cosine(pair, 200, "rank discount"),
            BEST: rare_words.rarity_cosine(pair, 5000),
            "8c rarity x probability cosine, top 5,000": rare_words.rarity_cosine(pair, 5000, "p")}


def rank_biased(pair: Pair) -> dict[str, np.ndarray]:
    return {f"3  rank-biased overlap, top {rank_weighted.TOP}, p = {rank_weighted.P}": rank_weighted.rank_biased_overlap(pair)}


def meanings(pair: Pair) -> dict[str, np.ndarray]:
    """No table without `pair.emb` (the embedding model)."""
    return {} if pair.emb is None else {f"10 meaning: probability-weighted centroid, top {meaning.TOP}": meaning.centroid_cosine(pair)}


def concept_scores(pair: Pair) -> dict[str, np.ndarray]:
    top = concept_lift.TOP
    return {f"11 concept score: log-probability above the control words, top {top:,}": concept_score.above_control_cosine(pair),
            f"12 unusual here: log-probability above the word's usual level, top {top:,}": concept_score.unusual_here_cosine(pair)}


SELECTIONS = {
    "all": (overlap, dot_products, rarity, min_and_max, rank_biased, meanings),
    "rare_words": (rarity,),
    "hybrids": (rarity_hybrids,),
    "concept_score": (concept_scores,),
}


def tables(pair: Pair, selection: str = "all") -> dict[str, np.ndarray]:
    """label -> table, for every method of the chosen selection."""
    out: dict[str, np.ndarray] = {}
    for family in SELECTIONS[selection]:
        out |= family(pair)
    return out
