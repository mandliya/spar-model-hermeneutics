"""Why do the J-space and NLA lie detectors fall to chance on a kind of lie they were not trained on?

    cd tianq && READOUT_BUNDLES=dyl-probe-repro/scratch/internal-readout-gap/bundles \
        python -m experiments.internal_readout_gap.lie_detection_gap.unseen_type_diagnosis --target gemma-3-27b-it
    (the dyl-probe-repro venv; reads the bundles and rows gap.py reads; CPU only, a few minutes)

One model reading its own conversations (no substitute), at the token before its answer to "did you lie?", label =
did the tested reply lie. Four ways of reading a row, all gap.py's: the probe on the activation, J-lens words, NLA
words, and words only (the conversation's text, no model). Six measurements, each a candidate cause:

  pairs      Train on one kind of lie, test on another, for every pair of kinds. Is it every pair that fails, or some?
             An AUROC well below 0.5 means the classifier's sign is flipped on the new kind.
  agreement  Does the same thing mark a lie in two kinds? Probe: cosine between two kinds' lie directions (mean lie
             activation minus mean truthful one, activations standardised over all rows). Words: correlation between
             two kinds' word log-odds (how much more often a word shows up under a lie), over the words both kinds
             use. Each is set beside the same number for two random halves of ONE kind, the most agreement the
             amount of data allows.
  topic      How well does each reading tell which kind of lie the row is (5 folds)? A reading that names the kind
             perfectly is mostly about the scenario.
  answer     Does the reading track what the model is about to SAY ("yes, I lied" or "no") rather than whether it
             lied? Same classifiers, trained to predict the model's own answer on kinds they have not seen.
  from_probe How much of the probe's score can the readout's words reproduce (ridge regression, 5 folds, R^2, inside
             one kind)? Low means the direction the probe uses is not what the readout puts into words.
  fixes      Unseen-kind AUROC again after three changes that use no label of the tested kind and no hand-picked word:
               centred       every kind's features have that kind's own mean removed (the tested kind's too)
               shared words  only words used in at least half of the training kinds
               not in text   a readout word is dropped when the row's own conversation contains it

Prints every number it writes; writes scratch/internal-readout-gap/unseen_type_diagnosis_<target>.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from dyl_repro.core import training
from experiments.cross_category_transfer.generalization import CATEGORIES
from experiments.hermeneutics import faithfulness
from experiments.internal_readout_gap.lie_detection_gap import gap

POSITION = gap.POSITIONS[0]
LABEL = "reply_lied"
WORD_READINGS = ("J-lens words", "NLA words")
READINGS = ("probe", *WORD_READINGS, "words only")
SEED = 0
TOKEN = re.compile(r"(?u)\b\w+\b")


def word_matrix(train: list, test: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """gap.py's features for a list of words, as dense matrices, and the vocabulary."""
    vec = gap.word_features()
    return vec.fit_transform(gap.readouts(train)).toarray(), vec.transform(gap.readouts(test)).toarray(), np.array(vec.words)


def fitted_features(reading: str, f: list):
    """Rows as the feature matrix their classifier is trained on: gap.py's for a readout's words, the arena's for
    the conversation's own words."""
    if reading == "words only":
        return faithfulness.words_vectorizer().fit_transform(f)
    return gap.word_features().fit_transform(gap.readouts(f))


def auroc(y, s) -> float:
    return faithfulness.auroc(np.asarray(y), np.asarray(s))


def enough(rows: list[dict], label: str) -> dict[str, list[int]]:
    """kind -> its rows with that label, for the kinds with at least 10 rows of each class (gap.py's rule)."""
    by = {c: [i for i, r in enumerate(rows) if r["category"] == c and r[label] is not None] for c in CATEGORIES}
    return {c: idx for c, idx in by.items()
            if idx and min(sum(rows[i][label] == v for i in idx) for v in (0, 1)) >= training.MIN_ROWS_PER_CLASS}


def within(reading: str, feats: list, y: np.ndarray) -> np.ndarray:
    """Out-of-fold scores inside one kind (5 stratified folds, seed 0)."""
    s = np.empty(len(y))
    for tr, te in StratifiedKFold(training.CROSS_VALIDATION_FOLDS, shuffle=True, random_state=SEED).split(feats, y):
        s[te] = gap.fit(reading, [feats[i] for i in tr], y[tr], None)([feats[i] for i in te])
    return s


def pairs(rows, feats, kinds) -> dict:
    out = {}
    for reading in READINGS:
        table = {}
        for a, b in itertools.product(kinds, kinds):
            ya, yb = (np.array([rows[i][LABEL] for i in kinds[k]]) for k in (a, b))
            fa, fb = ([feats[reading][i] for i in kinds[k]] for k in (a, b))
            table[f"{a} -> {b}"] = auroc(ya, within(reading, fa, ya)) if a == b else auroc(yb, gap.fit(reading, fa, ya, None)(fb))
        same = [v for k, v in table.items() if k.split(" -> ")[0] == k.split(" -> ")[1]]
        other = [v for k, v in table.items() if k.split(" -> ")[0] != k.split(" -> ")[1]]
        out[reading] = {"same_kind_mean": float(np.mean(same)), "other_kind_mean": float(np.mean(other)),
                        "other_kind_share_below_0.45": float(np.mean(np.array(other) < 0.45)),
                        "other_kind_share_above_0.60": float(np.mean(np.array(other) > 0.60)), "table": table}
        o = out[reading]
        print(f"pairs | {reading}: same kind {o['same_kind_mean']:.3f}; another kind {o['other_kind_mean']:.3f} over "
              f"{len(other)} pairs, {o['other_kind_share_above_0.60']:.0%} above 0.60, {o['other_kind_share_below_0.45']:.0%} "
              f"below 0.45 (sign flipped)", flush=True)
    return out


def log_odds(docs: list, y: np.ndarray, min_rows: int = 5) -> dict[str, float]:
    """word -> log odds of appearing in a lie row against a truthful row (0.5 added to each count), for words in at least `min_rows` rows."""
    lie, truth = (sum(y == v) for v in (1, 0))
    counts: dict[str, list[int]] = {}
    for doc, label in zip(docs, y):
        for w in set(doc):
            counts.setdefault(w, [0, 0])[int(label)] += 1
    return {w: float(np.log((c[1] + 0.5) / (lie + 1)) - np.log((c[0] + 0.5) / (truth + 1)))
            for w, c in counts.items() if sum(c) >= min_rows}


def agreement(rows, feats, kinds) -> dict:
    rng = np.random.default_rng(SEED)
    x = np.stack(feats["probe"])
    z = (x - x.mean(0)) / x.std(0)
    y_of = {k: np.array([rows[i][LABEL] for i in idx]) for k, idx in kinds.items()}

    def direction(idx, y):
        d = z[idx][y == 1].mean(0) - z[idx][y == 0].mean(0)
        return d / np.linalg.norm(d)

    def halves(k):
        order = rng.permutation(len(kinds[k]))
        cut = [order[::2], order[1::2]]
        return [([kinds[k][i] for i in h], y_of[k][h]) for h in cut]

    out = {"probe": {"between": [], "halves": []}} | {r: {"between": [], "halves": [], "shared_words": []} for r in WORD_READINGS}
    for k in kinds:
        (ia, ya), (ib, yb) = halves(k)
        if min(ya.sum(), yb.sum(), (ya == 0).sum(), (yb == 0).sum()) < 3:
            continue
        out["probe"]["halves"].append(float(direction(ia, ya) @ direction(ib, yb)))
        for r in WORD_READINGS:
            la, lb = log_odds([feats[r][i] for i in ia], ya, 3), log_odds([feats[r][i] for i in ib], yb, 3)
            common = sorted(set(la) & set(lb))
            if len(common) >= 20:
                out[r]["halves"].append(float(np.corrcoef([la[w] for w in common], [lb[w] for w in common])[0, 1]))
    for a, b in itertools.combinations(kinds, 2):
        out["probe"]["between"].append(float(direction(kinds[a], y_of[a]) @ direction(kinds[b], y_of[b])))
        for r in WORD_READINGS:
            la, lb = (log_odds([feats[r][i] for i in kinds[k]], y_of[k]) for k in (a, b))
            common = sorted(set(la) & set(lb))
            out[r]["shared_words"].append(len(common) / len(set(la) | set(lb)))
            if len(common) >= 20:
                out[r]["between"].append(float(np.corrcoef([la[w] for w in common], [lb[w] for w in common])[0, 1]))
    summary = {}
    for r, v in out.items():
        summary[r] = {k: float(np.mean(vals)) for k, vals in v.items() if vals} | {"pairs": len(v["between"])}
        s = summary[r]
        extra = f"; the two kinds share {s['shared_words']:.0%} of their words" if "shared_words" in s else ""
        print(f"agreement | {r}: between two kinds {s['between']:+.3f} (mean over {s['pairs']} pairs); between two halves of "
              f"one kind {s['halves']:+.3f}{extra}", flush=True)
    return summary


def topic(rows, feats, kinds) -> dict:
    idx = [i for k in kinds for i in kinds[k]]
    y = np.array([list(kinds).index(rows[i]["category"]) for i in idx])
    out = {"largest_kind_share": float(np.bincount(y).max() / len(y))}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for reading in READINGS:
        f = [feats[reading][i] for i in idx]
        if reading == "probe":
            x = np.stack(f)
            x = (x - x.mean(0)) / x.std(0)
        else:
            x = fitted_features(reading, f)
        pred = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), x, y, cv=cv)
        out[reading] = float((pred == y).mean())
        print(f"topic | {reading}: names the kind of lie in {out[reading]:.1%} of rows ({len(kinds)} kinds; always guessing "
              f"the largest gives {out['largest_kind_share']:.1%})", flush=True)
    return out


def unseen(rows, feats_of_reading: list, reading: str, label: str, kinds, prepare=None) -> dict[str, float]:
    """kind -> AUROC on that kind of a classifier trained on the other kinds, each weighted equally (gap.py's split).
    `prepare(train rows, test rows, feats)` may return changed (train features, test features); with it the
    classifier is plain logistic regression on what it returns."""
    out = {}
    for c in kinds:
        train = [i for k, idx in kinds.items() if k != c for i in idx]
        weight = np.concatenate([np.full(len(idx), 1 / len(idx)) for k, idx in kinds.items() if k != c])
        test = kinds[c]
        y = np.array([rows[i][label] for i in train])
        if prepare is None:
            s = gap.fit(reading, [feats_of_reading[i] for i in train], y, weight)([feats_of_reading[i] for i in test])
        else:
            xtr, xte = prepare(train, test, feats_of_reading)
            clf = LogisticRegression(C=1.0, max_iter=3000).fit(xtr, y, sample_weight=weight * len(weight))
            s = clf.decision_function(xte)
        out[c] = auroc([rows[i][label] for i in test], s)
    return out


def answer(rows, feats) -> dict:
    """said_yes = the model's answer to "did you lie?" was yes: it lied and did not deny it, or it did not lie and said it did."""
    for r in rows:
        r["said_yes"] = None if r["reply_lied"] is None or r["followup_lie"] is None else int(r["reply_lied"] != r["followup_lie"])
    kinds = enough(rows, "said_yes")
    out = {"kinds": list(kinds)}
    just_asking = [auroc([rows[i]["reply_lied"] for i in idx], [rows[i]["said_yes"] for i in idx]) for idx in enough(rows, LABEL).values()
                   if all(rows[i]["said_yes"] is not None for i in idx)]
    out["just_asking_auroc"] = float(np.mean(just_asking)) if just_asking else None
    for reading in READINGS:
        per = unseen(rows, feats[reading], reading, "said_yes", kinds)
        out[reading] = {"unseen_kind_mean": float(np.mean(list(per.values()))), "per_kind": per}
        print(f"answer | {reading}: predicts what the model is about to answer, on kinds it has not seen, AUROC "
              f"{out[reading]['unseen_kind_mean']:.3f} (mean over {len(per)} kinds: {', '.join(per)})", flush=True)
    print(f"answer | just asking: the model's own yes or no against whether it lied, AUROC {out['just_asking_auroc']}", flush=True)
    return out


def from_probe(rows, feats, kinds) -> dict:
    out = {r: [] for r in (*WORD_READINGS, "words only")}
    for k, idx in kinds.items():
        y = np.array([rows[i][LABEL] for i in idx])
        target = within("probe", [feats["probe"][i] for i in idx], y)
        target = (target - target.mean()) / target.std()
        for r in out:
            f = [feats[r][i] for i in idx]
            pred = cross_val_predict(Ridge(alpha=1.0), fitted_features(r, f), target, cv=5)
            out[r].append(float(1 - ((target - pred) ** 2).mean()))
    summary = {r: {"mean_r2": float(np.mean(v)), "per_kind": dict(zip(kinds, v))} for r, v in out.items()}
    for r, v in summary.items():
        print(f"from_probe | {r}: reproduces {v['mean_r2']:.1%} of the variance of the probe's score inside one kind "
              f"(mean over {len(kinds)} kinds)", flush=True)
    return summary


def fixes(rows, feats, kinds, text: list[str]) -> dict:
    kind_of = [r["category"] for r in rows]

    def centred(matrix_of):
        def prepare(train, test, f):
            xtr, xte = matrix_of(train, test, f)
            for k in {kind_of[i] for i in train}:
                m = np.array([kind_of[i] == k for i in train])
                xtr[m] -= xtr[m].mean(0)
            return xtr, xte - xte.mean(0)
        return prepare

    def words(train, test, f):
        return word_matrix([f[i] for i in train], [f[i] for i in test])[:2]

    def shared(train, test, f):
        xtr, xte, vocab = word_matrix([f[i] for i in train], [f[i] for i in test])
        train_kinds = sorted({kind_of[i] for i in train})
        used = np.stack([(xtr[np.array([kind_of[i] == k for i in train])] > 0).sum(0) >= 2 for k in train_kinds])
        keep = used.sum(0) >= len(train_kinds) / 2
        return xtr[:, keep], xte[:, keep]

    def activations(train, test, f):
        x = np.stack(f)
        z = (x - x[train].mean(0)) / x[train].std(0)
        return z[train].copy(), z[test].copy()

    own_words = [set(TOKEN.findall(t.lower())) for t in text]
    not_in_text = {r: [[w for w in doc if w.lower() not in own_words[i]] for i, doc in enumerate(feats[r])] for r in WORD_READINGS}
    for r in WORD_READINGS:
        kept = np.mean([len(a) / max(len(b), 1) for a, b in zip(not_in_text[r], feats[r])])
        print(f"fixes | {r}: {1 - kept:.1%} of a row's readout words also appear in its own conversation", flush=True)

    plans = {"probe": {"as it is": (feats["probe"], None), "centred": (feats["probe"], centred(activations))}}
    for r in WORD_READINGS:
        plans[r] = {"as it is": (feats[r], None), "centred": (feats[r], centred(words)), "shared words": (feats[r], shared),
                    "shared words, centred": (feats[r], centred(shared)), "not in text": (not_in_text[r], None),
                    "not in text, centred": (not_in_text[r], centred(words))}
    plans["words only"] = {"as it is": (feats["words only"], None)}
    out = {}
    for reading, versions in plans.items():
        out[reading] = {}
        for name, (f, prepare) in versions.items():
            per = unseen(rows, f, reading, LABEL, kinds, prepare)
            out[reading][name] = {"unseen_kind_mean": float(np.mean(list(per.values()))), "per_kind": per}
            print(f"fixes | {reading} | {name}: unseen kind AUROC {out[reading][name]['unseen_kind_mean']:.3f} "
                  f"({', '.join(f'{k} {v:.2f}' for k, v in per.items())})", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    args = ap.parse_args()
    read = gap.load_read(args.target, args.target)
    rows = [json.loads(line) for line in open(gap.LOCAL / f"rows_{args.target}.jsonl", encoding="utf-8")]
    rows = sorted((r for r in rows if r["uid"] in read and all(m in read[r["uid"]] for m in WORD_READINGS)),
                  key=lambda r: (r["category"], r["uid"]))
    text = ["\n".join(f"{m['role']}: {m['content']}" for m in r["messages"][:-2]) for r in rows]   # the conversation before the follow-up
    feats = {m: [read[r["uid"]][m][POSITION] for r in rows] for m in ("probe", *WORD_READINGS)} | {"words only": text}
    kinds = enough(rows, LABEL)
    print(f"rows | {args.target} reading itself at {POSITION}: {len(rows)} rows; kinds with at least "
          f"{training.MIN_ROWS_PER_CLASS} lies and {training.MIN_ROWS_PER_CLASS} truthful replies: "
          + ", ".join(f"{k} ({sum(rows[i][LABEL] for i in idx)} lies, {sum(1 - rows[i][LABEL] for i in idx)} truthful)" for k, idx in kinds.items()), flush=True)
    result = {"target": args.target, "position": POSITION, "label": LABEL, "rows": len(rows),
              "kinds": {k: {"lies": int(sum(rows[i][LABEL] for i in idx)), "truthful": int(sum(1 - rows[i][LABEL] for i in idx))} for k, idx in kinds.items()}}
    for name, run in (("pairs", lambda: pairs(rows, feats, kinds)), ("agreement", lambda: agreement(rows, feats, kinds)),
                      ("topic", lambda: topic(rows, feats, kinds)), ("answer", lambda: answer(rows, feats)),
                      ("from_probe", lambda: from_probe(rows, feats, kinds)), ("fixes", lambda: fixes(rows, feats, kinds, text))):
        result[name] = run()
    path = gap.LOCAL / f"unseen_type_diagnosis_{args.target}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
