"""Rare words count more. Method 8 and its variants. The best method on the same-moment test is 8b at top 5,000.

A word's weight is log(lists / lists that have the word): a word in almost every list counts for little. Each list
becomes a vector of those weights (times `value`), and two lists are compared by cosine. This one function is used
for J-lens readouts and for NLA sentences.

The lists the weights are counted over are the pair's pool (pair.py). Every run passes all three models' readouts
(Qwen3.6, base gemma, the final checkpoint) at every restart point, so base gemma and the final checkpoint are
scored with the same weights. (Until 20 Nov the J-space runs counted over the two models being compared only.)

  8   top 200, every word in the list weighted by rarity alone
  8b  the same at the top 50 / 1,000 / 5,000 words
  8c  rarity x the word's probability, top 5,000
  8d  rarity x sqrt(probability), top 5,000           (hybrids: softer confidence weights than 8c)
  8e  rarity x a rank discount 1 / log2(rank + 1), top 5,000 and top 200
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .pair import Pair

OTHER_TOPS = (50, 1000, 5000)   # added after TQ asked whether the cut at 200 is what makes method 8 strong
BEST = "8b rarity-weighted cosine, top 5,000"


def rarity_cosine(pair: Pair, top: int | None, value: str = "one") -> np.ndarray:
    """`top`: how far down each list to go (None: all of it). `value`: what a word is worth before its rarity
    weight ("one": 1 if it is in the list; "p": its probability or count; see pair.py for the others)."""
    lists_with_word = pair.lists_with_word(top)
    rarity = sparse.diags(np.log(len(pair.pool) * pair.n / np.maximum(lists_with_word, 1)).astype(np.float32))
    Ai, Bi = (m @ rarity for m in pair.matrices(top, value))
    return np.asarray((Ai @ Bi.T).toarray() / np.maximum(
        np.sqrt(Ai.multiply(Ai).sum(1)) @ np.sqrt(Bi.multiply(Bi).sum(1)).T, 1e-12))


def tables(pair: Pair) -> dict[str, np.ndarray]:
    out = {"8  rarity-weighted cosine, top 200": rarity_cosine(pair, 200)}
    out |= {f"8b rarity-weighted cosine, top {k:,}": rarity_cosine(pair, k) for k in OTHER_TOPS}
    out["8c rarity x probability cosine, top 5,000"] = rarity_cosine(pair, 5000, "p")
    return out


def hybrid_tables(pair: Pair) -> dict[str, np.ndarray]:
    """Added 18 Nov after TQ asked for rarity combined with rank or confidence; 8b and 8c are repeated beside them."""
    return {"8d rarity x sqrt(probability), top 5,000": rarity_cosine(pair, 5000, "sqrt p"),
            "8e rarity x rank discount 1/log2(rank + 1), top 5,000": rarity_cosine(pair, 5000, "rank discount"),
            "8e rarity x rank discount 1/log2(rank + 1), top 200": rarity_cosine(pair, 200, "rank discount"),
            BEST: rarity_cosine(pair, 5000),
            "8c rarity x probability cosine, top 5,000": rarity_cosine(pair, 5000, "p")}
