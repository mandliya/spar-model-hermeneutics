"""Two checks on the top-200 word overlap, from the saved reads (17 Nov, after TQ's questions; exploratory).

    READOUT_OUT=<folder from shared/fetch.py> python rank_weighted_overlap.py

top words   what sits at the top of each model's J-lens list at each depth: the most common first word over the
            restart points, how often a special or blank token is in the top 5, and the model's own most common next
            token. Found: Qwen3.6's first lens word is <|endoftext|> at 77% of points a quarter in, 83% halfway, 24% at
            two-thirds and 0% from three-quarters on, while its own next token is never that.
weighting   the distillation result again with agreement near the top of the lists counting more: rank-biased
            overlap (readout_methods/likeness/rank_weighted.py) at two persistences, beside the plain top-200
            Jaccard (readout_methods/likeness/word_overlap.py). Same subtraction of chance (CHANCE_SPOTS other restart points, fixed by SEED)
            and the same pairing with base gemma as similarity.py; intervals from resampling restart points.
            Found: the late-layer move is about three times larger rank-weighted, and two-thirds stays flat.
Only the order of the top 200 was saved by read_spots.py, so the weights here are by rank, not by probability
(write_jlens_lists.py and similarity_methods.py go back to the probabilities).
"""

from __future__ import annotations

import collections
import pathlib
import sys

import numpy as np

EXPERIMENT = pathlib.Path(__file__).resolve().parents[2]   # internal_readout_gap/
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(EXPERIMENT / "distillation_similarity"))
from readout_methods.likeness import rank_weighted, word_overlap  # noqa: E402
from readout_methods.likeness.pair import Pair  # noqa: E402
from readout_methods.readout_type import Readouts  # noqa: E402
from shared import readout, words  # noqa: E402
from similarity import ORIGINAL, VERSIONS, lens_lists, load  # noqa: E402

BASE, FINAL = VERSIONS[0], VERSIONS[-1]
CHANCE_SPOTS, SEED = 20, 0
PERSISTENCE = (0.9, 0.98)


def as_readouts(lists: dict[str, list[str]], names: list[str]) -> Readouts:
    """read_spots.py saved the order of the top 200 words and not their probabilities, so every word's value is 1."""
    return [(lists[n], np.ones(len(lists[n]), dtype=np.float32)) for n in names]


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
    others = np.array([rng.choice([j for j in range(len(common)) if j != i], CHANCE_SPOTS, replace=False) for i in range(len(common))])
    measures = {f"plain top-{rank_weighted.TOP} Jaccard": lambda pair: word_overlap.jaccard(pair, rank_weighted.TOP)}
    measures |= {f"rank-biased overlap, p = {p}": (lambda pair, p=p: rank_weighted.rank_biased_overlap(pair, p)) for p in PERSISTENCE}
    for d in readout.DEPTHS:
        lists = {t: as_readouts(lens_lists(lens[t], d), common) for t in lens}
        pairs = {t: Pair(lists[ORIGINAL], lists[t]) for t in (BASE, FINAL)}
        for name, measure in measures.items():
            above = {}
            for t, pair in pairs.items():
                table = measure(pair)   # [restart point of Qwen3.6, restart point of the gemma]
                above[t] = np.diag(table) - table[np.arange(len(common))[:, None], others].mean(1)
            m, lo, hi = words.mean_ci(above[FINAL] - above[BASE])
            print(f"weighting | depth {d:.3f} | {name}: change from base gemma, above chance {m:+.4f} [{lo:+.4f}, {hi:+.4f}]; "
                  f"base gemma above chance {above[BASE].mean():+.4f}", flush=True)


if __name__ == "__main__":
    main()
