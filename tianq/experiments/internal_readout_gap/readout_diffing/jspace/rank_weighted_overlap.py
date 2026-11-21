"""Two checks on the top-200 word overlap, from the saved reads (17 Nov, after TQ's questions; exploratory).

    READOUT_OUT=<folder from shared/fetch.py> python rank_weighted_overlap.py

top words   what sits at the top of each model's J-lens list at each depth: the most common first word over the
            restart points, how often a special or blank token is in the top 5, and the model's own most common next
            token. Found: Qwen3.6's first lens word is <|endoftext|> at 77% of points a quarter in, 83% halfway, 24% at
            two-thirds and 0% from three-quarters on, while its own next token is never that.
weighting   the distillation result again with agreement near the top of the lists counting more: rank-biased
            overlap (Webber et al. 2010, truncated at the shorter list and normalized) at two persistences, beside
            the plain top-200 Jaccard. Same subtraction of chance (CHANCE_SPOTS other restart points, fixed by SEED)
            and the same pairing with base gemma as similarity.py; intervals from resampling restart points.
            Found: the late-layer move is about three times larger rank-weighted, and two-thirds stays flat.
Only the order of the top 200 was saved by read_spots.py, so the weights here are by rank, not by probability
(lens_distributions.py and similarity_methods.py go back to the probabilities).
"""

from __future__ import annotations

import collections
import pathlib
import sys

import numpy as np

EXPERIMENT = pathlib.Path(__file__).resolve().parents[2]   # internal_readout_gap/
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(EXPERIMENT / "distillation_similarity"))
from shared import readout, words  # noqa: E402
from similarity import ORIGINAL, VERSIONS, lens_lists, load  # noqa: E402

BASE, FINAL = VERSIONS[0], VERSIONS[-1]
CHANCE_SPOTS, SEED = 20, 0
PERSISTENCE = (0.9, 0.98)


def rbo(a: list[str], b: list[str], p: float) -> float:
    """Rank-biased overlap of two ranked lists: the agreement of their top-k sets, averaged over k with weight p^(k-1)."""
    seen_a, seen_b, overlap, total, norm = set(), set(), 0, 0.0, 0.0
    for k in range(1, min(len(a), len(b)) + 1):
        x, y = a[k - 1], b[k - 1]
        overlap += 1 if x == y else (x in seen_b) + (y in seen_a)
        seen_a.add(x)
        seen_b.add(y)
        total += p ** (k - 1) * overlap / k
        norm += p ** (k - 1)
    return total / norm if norm else float("nan")


def main() -> None:
    lens = {t: load(t, "lens") for t in (ORIGINAL, BASE, FINAL)}
    common = sorted(set.intersection(*(set(v) for v in lens.values())))
    print(f"{len(common)} restart points", flush=True)
    for t in (ORIGINAL, BASE):
        for d in readout.DEPTHS:
            lists = lens_lists(lens[t], d)
            first = collections.Counter(lists[n][0] for n in common)
            special = np.mean([any(w.startswith("<") or not w.strip() for w in lists[n][:5]) for n in common])
            print(f"top words | {t} | depth {d:.3f}: most common first word {[(w, c) for w, c in first.most_common(5)]}; "
                  f"a special or blank token in the top 5 at {special:.2f} of points")
        own = collections.Counter(words.lens_words(lens[t][n]["model"])[0] for n in common)
        print(f"top words | {t} | its own next token: {[(w, c) for w, c in own.most_common(6)]}")

    rng = np.random.default_rng(SEED)
    others = {n: [common[j] for j in rng.choice([j for j in range(len(common)) if common[j] != n], CHANCE_SPOTS, replace=False)]
              for n in common}
    measures = {"plain top-200 Jaccard": lambda a, b: words.jaccard(set(a), set(b))}
    measures |= {f"rank-biased overlap, p = {p}": (lambda a, b, p=p: rbo(a, b, p)) for p in PERSISTENCE}
    for d in readout.DEPTHS:
        lists = {t: lens_lists(lens[t], d) for t in lens}
        for name, measure in measures.items():
            above = {}
            for t in (BASE, FINAL):
                same = np.array([measure(lists[ORIGINAL][n], lists[t][n]) for n in common])
                chance = np.array([np.mean([measure(lists[ORIGINAL][n], lists[t][m]) for m in others[n]]) for n in common])
                above[t] = same - chance
            m, lo, hi = words.mean_ci(above[FINAL] - above[BASE])
            print(f"weighting | depth {d:.3f} | {name}: change from base gemma, above chance {m:+.4f} [{lo:+.4f}, {hi:+.4f}]; "
                  f"base gemma above chance {above[BASE].mean():+.4f}", flush=True)


if __name__ == "__main__":
    main()
