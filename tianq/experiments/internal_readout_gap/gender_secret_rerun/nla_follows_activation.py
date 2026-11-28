"""Do the NLA's sentences follow the activation they were written from? The check that stands in for the
shuffled-activation control ("Current Activation Oracles Are Hard to Use": a decoder that scored as well with the
activations swapped between rows, because it answered from the text it was also shown).

    cd tianq && PYTHONPATH=$PWD/dyl-probe-repro:$PWD python -m experiments.internal_readout_gap.gender_secret_rerun.nla_follows_activation

Our NLA is shown nothing but the activation (one fixed prompt, shared/nla_read.py), so swapping activations between
rows is the same as shuffling the labels: every monitor falls to chance by construction, and running it would show
nothing. What can fail is the step before: the five sentences may be the NLA's habits and not a reading of the
activation. So, for every read of the gender rows and each position: the five samples of a row are split into its
first two and its last three, and the first two of row i are compared with the last three of every row j (the
arena's NLA winner: word counts, rare words counting more).
  right row found   the share of rows whose own other half is the closest of all rows (chance: 1 in the row count)
  above other rows  likeness to the row's own other half minus its mean likeness to other rows' (0 = no dependence)
Prints what it writes; writes scratch/.../gender_secret_rerun/nla_follows_activation.json.
"""

from __future__ import annotations

import json

import numpy as np

from experiments.internal_readout_gap.gender_secret_rerun import gap, transcript
from experiments.internal_readout_gap.gender_secret_rerun.rows import DISTILLED
from experiments.internal_readout_gap.readout_methods import readout_type
from experiments.internal_readout_gap.readout_methods.likeness import evaluation, rare_words
from experiments.internal_readout_gap.readout_methods.likeness.pair import Pair
from experiments.internal_readout_gap.shared import words

FIRST = 2   # samples in the first half; the rest are the second


def main() -> None:
    result = {}
    for model in (gap.AUTHOR, gap.READER, DISTILLED):
        for condition in transcript.CONDITIONS:
            read = gap.load_read("gender_secret", condition, model)
            for position in transcript.POSITIONS:
                sentences = read[position]["NLA sentences"]
                uids = sorted(u for u, s in sentences.items() if any(x.strip() for x in s[:FIRST]) and any(x.strip() for x in s[FIRST:]))
                halves = [[readout_type.from_words(words.text_words(part(sentences[u]))) for u in uids] for part in (lambda s: s[:FIRST], lambda s: s[FIRST:])]
                table = rare_words.rarity_cosine(Pair(*halves), top=None, value="p")
                found = float((table.argmax(1) == np.arange(len(uids))).mean())
                m, lo, hi = words.mean_ci(evaluation.above_chance(table))
                result[f"{model} | {condition} | {position}"] = {"right_row_found": found, "chance": 1 / len(uids),
                                                                 "above_other_rows": m, "ci": [lo, hi], "rows": len(uids)}
                print(f"{model} | {condition} | {position}: right row found for {found:.1%} of {len(uids)} rows (chance "
                      f"{1 / len(uids):.1%}); above other rows {m:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
    path = gap.LOCAL / "nla_follows_activation.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
