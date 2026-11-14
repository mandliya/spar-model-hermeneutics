"""Question 1 again, graded by a judge model: is the distilled gemma's readout closer to Qwen3.6's in meaning?

    READOUT_OUT=<folder from shared/fetch.py> python judge_similarity.py --limit 3     # test on 3 spots
    READOUT_OUT=<folder from shared/fetch.py> python judge_similarity.py               # every spot (resumable)

A judge model (shared.judge: OpenRouter, provider pinned, no fallbacks) reads two readouts, not told which model or spot they come
from, and rates how alike they are in meaning, 0 to 10. For each gemma version and each spot it rates two pairs:
Qwen3.6's readout with the version's at the same spot, and Qwen3.6's with the version's at one other spot (chance;
the other spot is fixed per spot by a seeded shuffle, the same for every version). Three kinds of readout:
  NLA sentences                    the spot's explanations (up to 5)
  J-lens words | depth 0.667       the top TOP_WORDS J-lens words two-thirds of the way in (the headline depth)
  J-lens words | depth 0.950       the same, near the output
Summaries are similarity.py's: same spot, chance, above chance, and the paired change from base gemma, per spot group,
with 95% intervals from resampling spots. Answers are cached one per line in OUT/distillation_similarity/judge_similarity/,
so a rerun only asks what is missing; it stops before the total cost passes BUDGET.
Added on 13 Nov with meaning_similarity.py, to test whether "no move toward Qwen3.6 in the middle layers" is the
fault of counting identical words. Prints every number it writes; writes OUT/distillation_similarity/judge_similarity.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import models, readout  # noqa: E402
from shared.judge import JUDGE, READOUT_KINDS, Judge  # noqa: E402
from similarity import EXPERIMENT, ORIGINAL, VERSIONS, grouped, lens_lists, load, spot_groups  # noqa: E402

BUDGET = 15.0
TOP_WORDS = 50
SEED = 0
PROMPT = """Two interpretability tools each produced a readout of what an AI agent's internal state was about at some \
moment. Readout A and readout B may come from different models and different moments. {what}

<readout_A>
{a}
</readout_A>

<readout_B>
{b}
</readout_B>

How similar are the two readouts in meaning? Judge what they are about (the situation, the action about to be taken, \
the attitude), not their wording, language or length: synonyms and translations of the same word count as the same. \
0 = unrelated, 5 = partly the same subject, 10 = the same meaning. Answer with one whole number from 0 to 10 and nothing else."""


def readouts(kind: str, lens: dict, nla: dict, names: list[str]) -> dict[str, dict[str, str]]:
    """tag -> spot -> the text the judge sees for this kind of readout."""
    if kind == "NLA sentences":
        return {t: {n: "\n".join(f"- {s['explanation'].strip()}" for s in nla[t][n]["positions"]["before_action"]
                                 if s["explanation"].strip()) for n in names} for t in nla}
    depth = float(kind.split("depth ")[1])
    return {t: {n: ", ".join(w[:TOP_WORDS]) for n, w in lens_lists(lens[t], depth).items() if n in names} for t in lens}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N spots (the test)")
    args = ap.parse_args()
    tags = [ORIGINAL, *VERSIONS]
    lens = {t: load(t, "lens") for t in tags}
    nla = {t: load(t, "nla") for t in tags}
    common = sorted(set.intersection(*(set(lens[t]) & set(nla[t]) for t in tags)))
    order = np.random.default_rng(SEED).permutation(len(common))
    other = {common[i]: common[j] for i, j in zip(order, np.roll(order, 1))}   # one different spot for every spot
    names = common[: args.limit]
    groups = spot_groups(names)
    judge = Judge(models.OUT / EXPERIMENT / "judge_similarity", BUDGET)
    print(f"judge {JUDGE}: {len(names)} spots, {len(VERSIONS)} versions, 2 pairs each, 3 kinds of readout; "
          f"${judge.spent:.2f} already spent, {len(judge.cache)} answers cached", flush=True)
    result: dict = {"judge": JUDGE, "spots": len(names), "measures": {}}
    for kind in ("NLA sentences", f"J-lens words | depth {readout.HEADLINE:.3f}", "J-lens words | depth 0.950"):
        text = readouts(kind, lens, nla, common)
        what = READOUT_KINDS[kind.split(" | ")[0]]
        base = None
        for v in VERSIONS:
            pairs = [(n, pairing, text[ORIGINAL][n], text[v][n if pairing == "same" else other[n]])
                     for n in names for pairing in ("same", "other")]
            ratings = judge.rate_all([(f"{kind} | {v} || {n} || {pairing}", PROMPT.format(what=what, a=a, b=b) if a and b else None)
                                      for n, pairing, a, b in pairs])
            same, chance = np.array(ratings[0::2]), np.array(ratings[1::2])
            unparsed = int(sum(1 for p, r in zip(pairs, ratings) if np.isnan(r) and p[2] and p[3]))
            print(f"judge | {kind} | {v}: {len(pairs)} pairs rated, {unparsed} answers not a number; spent ${judge.spent:.2f}", flush=True)
            result["measures"].setdefault(f"judge | {kind}", {})[v] = grouped(f"judge | {kind}", v, names, groups, same, chance, base)
            base = base or (same, chance)
    result["spent"] = judge.spent
    path = models.OUT / EXPERIMENT / "judge_similarity.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"judge spend ${judge.spent:.2f}; wrote {path}", flush=True)


if __name__ == "__main__":
    main()
