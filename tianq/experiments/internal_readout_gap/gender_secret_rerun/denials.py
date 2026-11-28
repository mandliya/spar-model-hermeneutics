"""What J space and NLA show at the organism's denials (PLAN.md, step 1), from the gender rows alone: nothing is trained.

    cd tianq && PYTHONPATH=$PWD/dyl-probe-repro:$PWD python -m experiments.internal_readout_gap.gender_secret_rerun.denials
    ... denials --reader <tag>      # another reader in place of the untouched model
    (the dyl-probe-repro venv; CPU; reads what bundle.py and write_jlens_lists.py uploaded to HF)

For the organism reading itself and for the reader reading the organism's transcript, in both conditions and at both
positions, for the J-lens words (two thirds of the way in) and the NLA sentences:
  female words   the share of lie rows and of honest rows whose readout has a female word (J space: among its first
                 SHORT words), and the same for male words
  score          female minus male (gap.untrained), as an AUROC of lie against honest rows, and the gap between the
                 organism and the reader with the usual interval
  telling words  the words most tied to lie rows and to honest rows (how much more often a word is in one class's
                 readouts, as a log odds ratio with one added to every count; words in fewer than MIN_ROWS rows left out)
  examples       the first EXAMPLES lie rows: the question, and each model's first words or first sentence
And once per condition, from no readout at all: the share of lie rows and of honest rows in which what the model
itself wrote before its yes or no (its own reasoning about "Did you lie?") has a female word. This is the check on
what the token before the yes or no can be read from. It is an observation, not a baseline: the words-only baseline
is never given reasoning.
Prints everything; writes scratch/internal-readout-gap/gender_secret_rerun/denials_<author>__<reader>.json.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from experiments.hermeneutics import faithfulness
from experiments.internal_readout_gap.gender_secret_rerun import gap, transcript
from experiments.internal_readout_gap.shared import words

SHORT, MIN_ROWS, TELLING, EXAMPLES = 200, 5, 15, 3


def words_of(source: str, feature) -> list[str]:
    return feature[0][:SHORT] if source == "J space" else sorted(set(feature))


def telling(rows_words: dict[str, list[str]], labels: dict[str, int]) -> dict:
    """The TELLING words most tied to lie rows and to honest rows, and for each of them how many rows of each class
    have it. A label is 1 for a row where the MO's answer was a lie and 0 for an honest one (rows.py)."""
    honest, lie = 0, 1
    n = {c: sum(labels[u] == c for u in rows_words) for c in (honest, lie)}
    have = {}   # word -> [honest rows with it, lie rows with it]
    for u, ws in rows_words.items():
        for w in set(ws):
            have.setdefault(w, [0, 0])[labels[u]] += 1
    odds = {w: math.log((c[lie] + 1) / (n[lie] - c[lie] + 1)) - math.log((c[honest] + 1) / (n[honest] - c[honest] + 1))
            for w, c in have.items() if sum(c) >= MIN_ROWS}
    ranked = sorted(odds, key=odds.get)
    listed = {"lie rows": ranked[::-1][:TELLING], "honest rows": ranked[:TELLING]}
    return listed | {"rows with the word": {w: {"lie": have[w][lie], "honest": have[w][honest]} for ws in listed.values() for w in ws},
                     "rows": {"lie": n[lie], "honest": n[honest]}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default=gap.READER)
    args = ap.parse_args()
    gender = gap.load_rows("gender_secret")
    labels = {u: r["reply_lied"] for u, r in gender.items() if r["reply_lied"] is not None}
    reads = {who: {c: gap.load_read("gender_secret", c, model) for c in transcript.CONDITIONS}
             for who, model in (("organism", gap.AUTHOR), ("reader", args.reader))}
    result: dict = {}
    for condition in transcript.CONDITIONS:
        named = {}
        for who in reads:
            own = reads[who][condition]["before_answer"]["own reasoning"]
            named[who] = {f"female word in {cls} rows": float(np.mean([bool(gap.FEMALE & set(words.text_words([own[u]])))
                                                                      for u in own if labels.get(u) == c]))
                          for cls, c in (("lie", 1), ("honest", 0))}
            print(f"{condition} | own reasoning before the yes or no | {who}: a female word in "
                  f"{named[who]['female word in lie rows']:.0%} of lie rows and {named[who]['female word in honest rows']:.0%} of honest rows", flush=True)
        result[f"{condition} | own reasoning before the yes or no"] = named
        for position in transcript.POSITIONS:
            for source in ("J space", "NLA"):
                tag = f"{condition} | {position} | {source}"
                feats = {who: reads[who][condition][position].get(source) for who in reads}
                if any(f is None for f in feats.values()):
                    print(f"{tag}: not computed, a model has no such readout yet", flush=True)
                    continue
                uids = sorted(set(feats["organism"]) & set(feats["reader"]) & set(labels))
                y = np.array([labels[u] for u in uids])
                entry: dict = {"rows": len(uids)}
                scores = {}
                for who, f in feats.items():
                    listed = {u: words_of(source, f[u]) for u in uids}
                    share = {f"{name} word in {cls} rows": float(np.mean([bool(group & set(listed[u])) for u in uids if labels[u] == c]))
                             for name, group in (("female", gap.FEMALE), ("male", gap.MALE)) for cls, c in (("lie", 1), ("honest", 0))}
                    scores[who] = {"uids": uids, "label": y.tolist(), "score": [gap.untrained(source, f[u]) for u in uids]}
                    entry[who] = {"shares": share, "telling_words": telling(listed, labels)}
                    print(f"{tag} | {who}: a female word in {share['female word in lie rows']:.0%} of lie rows and "
                          f"{share['female word in honest rows']:.0%} of honest rows; a male word in {share['male word in lie rows']:.0%} "
                          f"and {share['male word in honest rows']:.0%}", flush=True)
                    print(f"{tag} | {who} | words tied to lie rows: {entry[who]['telling_words']['lie rows']}", flush=True)
                    print(f"{tag} | {who} | words tied to honest rows: {entry[who]['telling_words']['honest rows']}", flush=True)
                g = faithfulness.gap(scores["organism"], scores["reader"])
                entry["female_minus_male"] = g
                print(f"{tag} | female minus male, no training: organism {g['target']:.3f}, reader {g['substitute']:.3f}, gap "
                      f"{g['gap']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}] ({g['n_lie']} lie, {g['n_honest']} honest rows)", flush=True)
                for u in [u for u in uids if labels[u] == 1][:EXAMPLES]:
                    print(f"{tag} | example {u}: {gender[u]['transcript'][0]['content']!r}", flush=True)
                    for who, f in feats.items():
                        shown = f[u][0][:25] if source == "J space" else reads[who][condition][position]["NLA sentences"][u][:1]
                        print(f"    {who}: {shown}", flush=True)
                result[tag] = entry
    path = gap.LOCAL / f"denials_{gap.AUTHOR}__{args.reader}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
