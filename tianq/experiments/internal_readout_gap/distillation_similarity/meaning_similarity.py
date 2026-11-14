"""Question 1 again, by meaning instead of identical words: is the distilled gemma's readout closer to Qwen3.6's?

    READOUT_OUT=<folder from shared/fetch.py> HF_HOME=<your HF cache> python meaning_similarity.py

similarity.py counts identical words, so "retry" against "trying", or "now" against its Russian, score nothing. Here
every word and every NLA explanation is turned into a meaning vector (shared.embedding), and readouts are compared
by cosine:
  NLA meaning          each spot's explanations averaged into one vector; cosine between the two models' vectors
  J-lens centroid      the J-lens top-200 words averaged into one vector; cosine between the two models' vectors
  J-lens soft match    for each word of one list its best-matching word in the other (cosine), averaged both ways:
                       two lists of near-synonyms or translations score high without sharing a word
Everything else is similarity.py's: the same spots and groups, the same chance level (the original at one spot against
the version at other spots; all other spots, or CHANCE_SPOTS random ones for the slower soft match), the paired change
from base gemma, and 95% intervals from resampling spots. J-lens depths: 2/3 (the headline) and 0.95.
Added on 13 Nov after the word-overlap results, to test whether "no move toward Qwen3.6 in the middle layers" is the
measure's fault. Prints every number it writes; writes OUT/distillation_similarity/meaning_similarity.json.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import models, readout  # noqa: E402
from shared.embedding import EMBEDDER, Embedder  # noqa: E402
from similarity import EXPERIMENT, ORIGINAL, VERSIONS, grouped, lens_lists, load, spot_groups  # noqa: E402

DEPTHS = (readout.HEADLINE, 0.95)
CHANCE_SPOTS = 20
SEED = 0


def centroid_scores(orig: np.ndarray, vers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows are spots. Same-spot cosine, and the mean cosine against the version at every other spot."""
    sims = orig @ vers.T
    same = np.diag(sims)
    chance = (np.nansum(sims, axis=1) - np.nan_to_num(same)) / (np.sum(~np.isnan(sims), axis=1) - (~np.isnan(same)))
    return same, chance


def soft_match(a: np.ndarray, b: np.ndarray) -> float:
    sims = a @ b.T
    return float((sims.max(1).mean() + sims.max(0).mean()) / 2)


def soft_scores(orig: list[np.ndarray], vers: list[np.ndarray], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = len(orig)
    same = np.array([soft_match(orig[i], vers[i]) for i in range(n)])
    chance = np.array([np.mean([soft_match(orig[i], vers[j])
                                for j in rng.choice([j for j in range(n) if j != i], CHANCE_SPOTS, replace=False)])
                       for i in range(n)])
    return same, chance


def explanations(reads: dict[str, dict], name: str) -> list[str]:
    return [s["explanation"].strip() for s in reads[name]["positions"]["before_action"] if s["explanation"].strip()]


def main() -> None:
    tags = [ORIGINAL, *VERSIONS]
    lens = {t: load(t, "lens") for t in tags}
    nla = {t: load(t, "nla") for t in tags}
    common = sorted(set.intersection(*(set(lens[t]) & set(nla[t]) for t in tags)))
    groups = spot_groups(common)
    print(f"spots with a J-lens and an NLA readout for every model: {len(common)}; embedder {EMBEDDER}", flush=True)
    emb = Embedder()
    result: dict = {"embedder": EMBEDDER, "spots": len(common), "measures": {}}

    emb.add({e for t in tags for n in common for e in explanations(nla[t], n)})
    vectors = {t: np.stack([emb.mean(explanations(nla[t], n)) for n in common]) for t in tags}
    base = None
    for v in VERSIONS:
        same, chance = centroid_scores(vectors[ORIGINAL], vectors[v])
        result["measures"].setdefault("NLA meaning", {})[v] = grouped("NLA meaning", v, common, groups, same, chance, base)
        base = base or (same, chance)

    for depth in DEPTHS:
        lists = {t: lens_lists(lens[t], depth) for t in tags}
        emb.add({w for t in tags for n in common for w in lists[t][n]})
        per_spot = {t: [emb.matrix(lists[t][n]) for n in common] for t in tags}
        centroids = {t: np.stack([m.mean(0) / np.linalg.norm(m.mean(0)) for m in per_spot[t]]) for t in tags}
        for kind, score in (("J-lens centroid", lambda v: centroid_scores(centroids[ORIGINAL], centroids[v])),
                            ("J-lens soft match", lambda v: soft_scores(per_spot[ORIGINAL], per_spot[v],
                                                                        np.random.default_rng(SEED)))):
            name, base = f"{kind} | depth {depth:.3f}", None
            for v in VERSIONS:
                same, chance = score(v)
                result["measures"].setdefault(name, {})[v] = grouped(name, v, common, groups, same, chance, base)
                base = base or (same, chance)

    path = models.OUT / EXPERIMENT / "meaning_similarity.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
