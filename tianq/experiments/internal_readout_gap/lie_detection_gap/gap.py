"""Question 2: the faithfulness gap with J-space and NLA, on the lie-detection data (John's bag of words).

    cd tianq && python -m experiments.internal_readout_gap.lie_detection_gap.gap --target gemma-3-27b-it --substitute gemma-3-12b-it
    (the dyl-probe-repro venv, on a machine with the git repo; reads the bundles bundle.py made, from HF or, with
    READOUT_BUNDLES=<folder>, from copies of them on disk)

"the dumbest possible solution ... is to do bag of words ... that classifies, is this a dishonest rollout or an honest
rollout ... you train it on the target model and then you use it to test on the substitute model" (John, 10 Nov).

The target reads its own conversations; the substitute is prefilled with the same conversations. Each read gives, at
each of the two positions (before_answer, answer_token): the J-lens top-200 words two-thirds of the way in, the NLA
explanations where the model has an NLA, and the activation at the arena's probe layer. A method scores every row:
  J-lens words   TF-IDF over the readout's words (shared.words.lens_words), logistic regression
  NLA words      TF-IDF over the explanations' words (shared.words.text_words), logistic regression
  probe          the arena's probe recipe on the activation (dyl_repro.core.training.fit_weights), shown alongside
  words only     no model: TF-IDF over the conversation itself, the arena's baseline (experiments.hermeneutics.faithfulness)
For the two word methods the classifier is trained on the TARGET's readouts and scored on held-out target rows
("target"), then on the SUBSTITUTE's readouts of the same held-out rows ("substitute"). A second version gives the
substitute its own classifier ("own"), the only version a probe can have (activations of two models don't line up).
    gap = target AUROC - substitute AUROC, both against the target's labels
Labels (the arena's): "lie vs truthful" trains and scores on whether the tested reply was a lie; "doubled-down vs
truthful denial" trains on whether the follow-up answer was a lie and scores the denials only.
Splits: "same type" = within a category, 5 stratified folds (seed 0); "unseen type" = trained on the other
categories, each weighted equally, scored on the one left out ("training on one flavor of deception and then testing
on another flavor"). A category counts when it has at least 10 rows of each scored class (and of each training class).
Per category: AUROCs, the gap and its 95% interval (faithfulness.gap, paired class-stratified bootstrap). Over
categories: the mean, and a 95% interval for the mean gap from the same bootstrap run inside every category at once.
Fixed before any result: depth 2/3, top-200, before_answer as the headline position (the arena probe's own).
Prints every number it writes; writes scratch/internal-readout-gap/gap_<target>__<substitute>.json.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pathlib

import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from dyl_repro.core import training
from dyl_repro.core.store import SCRATCH
from experiments.cross_category_transfer.generalization import CATEGORIES, FEW
from experiments.hermeneutics import faithfulness
from experiments.internal_readout_gap.shared import readout, words

EXPERIMENT = "lie_detection_gap"
LOCAL = pathlib.Path(SCRATCH) / "internal-readout-gap"
POSITIONS = ("before_answer", "answer_token")
LABELS = {"lie vs truthful": ("reply_lied", "reply_lied"),                      # name: (training label, scored label)
          "doubled-down vs truthful denial": ("followup_lie", "doubled_down")}
SPLITS = ("same type", "unseen type")
WORD_METHODS = ("J-lens words", "NLA words")
MEANING_METHODS = ("J-lens meaning", "NLA meaning")   # meaning vectors, built and scored by meaning_gap.py
TOP_WORDS = 15


def bundle(read: str, name: str) -> str:
    """A read's bundle file: from HF, or from READOUT_BUNDLES/<read>/ when that names a folder of copies already on disk."""
    if os.environ.get("READOUT_BUNDLES"):
        return str(pathlib.Path(os.environ["READOUT_BUNDLES"]) / read / name)
    return hf_hub_download(os.environ["HF_DATASET_REPO"], f"internal-readout-gap/{EXPERIMENT}/outputs/{read}/{name}",
                           repo_type="dataset", local_dir=LOCAL / "hub", token=os.environ["HF_TOKEN"])


def load_read(target: str, model: str) -> dict:
    """uid -> method -> position -> features, for one read."""
    read = f"{target}__read_by_{model}"
    with gzip.open(bundle(read, "words.json.gz"), "rt", encoding="utf-8") as f:
        b = json.load(f)
    probe = np.load(bundle(read, "probe_layer.npz"))
    x = dict(zip(probe["uids"].tolist(), probe["x"]))
    layer = None
    out = {}
    for uid, row in b["rows"].items():
        layer = layer or str(readout.layer_at(b["n_layers"], [int(k) for k in row["lens"][POSITIONS[0]]], readout.HEADLINE))
        feats = {"J-lens words": {p: words.lens_words({"words": row["lens"][p][layer]}) for p in POSITIONS},
                 "probe": {p: x[uid][i] for i, p in enumerate(probe["positions"].tolist())}}
        if "nla" in row:
            feats["NLA words"] = {p: words.text_words(row["nla"][p]) for p in POSITIONS}
            feats["NLA sentences"] = {p: [e.strip() for e in row["nla"][p] if e.strip()] for p in POSITIONS}
        out[uid] = feats
    print(f"read | {read}: {len(out)} rows; J-lens layer {layer} of {b['n_layers']} blocks; probe layer {int(probe['layer'])}; "
          f"{sum('NLA words' in f for f in out.values())} rows with NLA sentences; recorded answer = the model's top "
          f"next token in {sum(r['answer_is_model_top'] for r in b['rows'].values())} rows", flush=True)
    return out


def fit(method: str, feats: list, y: np.ndarray, weight: np.ndarray | None):
    """Train one method; returns a function scoring a list of features."""
    if method == "probe" or method in MEANING_METHODS:   # vectors: the arena's probe recipe
        mean, std, w, b = training.fit_weights(np.stack(feats), y, training.DEFAULT_L2_LAMBDA, weight)
        return lambda test: ((np.stack(test) - mean) / std) @ w + b
    vec = faithfulness.words_vectorizer() if method == "words only" else TfidfVectorizer(
        analyzer=lambda doc: doc, min_df=2, sublinear_tf=True)
    clf = faithfulness.words_classifier().fit(vec.fit_transform(feats), y, sample_weight=weight)
    return lambda test: clf.decision_function(vec.transform(test))


def folds(rows: list[dict], train_label: str, split: str):
    """(category, training row indices, sample weights, row indices to score) for every fit of a split."""
    by_cat = {c: [i for i, r in enumerate(rows) if r["category"] == c and r[train_label] is not None] for c in CATEGORIES}
    enough = {c: idx for c, idx in by_cat.items()
              if min(sum(rows[i][train_label] == v for i in idx) for v in (0, 1)) >= training.MIN_ROWS_PER_CLASS}
    if split == "same type":
        for c, idx in enough.items():
            y = np.array([rows[i][train_label] for i in idx])
            for tr, te in StratifiedKFold(training.CROSS_VALIDATION_FOLDS, shuffle=True, random_state=0).split(idx, y):
                yield c, [idx[i] for i in tr], None, [idx[i] for i in te]
    else:
        for c in CATEGORIES:
            others = {o: idx for o, idx in enough.items() if o != c}
            train = [i for idx in others.values() for i in idx]
            weight = np.concatenate([np.full(len(idx), 1 / len(idx)) for idx in others.values()])
            yield c, train, weight, [i for i, r in enumerate(rows) if r["category"] == c]


def score(rows: list[dict], feats_t: list, feats_s: list | None, method: str, labels: tuple[str, str], split: str) -> dict:
    """Out-of-sample scores for every scored row: the target, the substitute through the target's classifier
    (not the probe: activations of two models don't line up), and the substitute through its own."""
    train_label, test_label = labels
    n = len(rows)
    s = {k: np.full(n, np.nan) for k in ("target", "substitute", "own")}
    for _, train, weight, test in folds(rows, train_label, split):
        y = np.array([rows[i][train_label] for i in train])
        scorer = fit(method, [feats_t[i] for i in train], y, weight)
        s["target"][test] = scorer([feats_t[i] for i in test])
        if feats_s is not None:
            if method != "probe":   # words and meaning vectors are the same space for both models
                s["substitute"][test] = scorer([feats_s[i] for i in test])
            s["own"][test] = fit(method, [feats_s[i] for i in train], y, weight)([feats_s[i] for i in test])
    out = {}
    for c in CATEGORIES:
        idx = [i for i, r in enumerate(rows) if r["category"] == c and r[test_label] is not None and not np.isnan(s["target"][i])]
        if idx:
            base = {"uids": [rows[i]["uid"] for i in idx], "label": [int(rows[i][test_label]) for i in idx]}
            out[c] = {k: base | {"score": v[idx].tolist()} for k, v in s.items() if not np.isnan(v[idx]).any()}
    return out


def mean_gap_ci(scored: dict, other: str) -> list[float] | None:
    """95% interval for the mean gap over trusted categories: faithfulness.gap's bootstrap, run in every category at once."""
    kept = []
    for c in scored.values():
        y = np.array(c["target"]["label"])
        if other in c and min(int(y.sum()), int((y == 0).sum())) >= FEW:
            kept.append((y, np.array(c["target"]["score"]), np.array(c[other]["score"])))
    if not kept:
        return None
    rng = np.random.default_rng(faithfulness.SEED)
    means = np.empty(faithfulness.BOOTSTRAP)
    for b in range(faithfulness.BOOTSTRAP):
        gaps = []
        for y, a, s in kept:
            lies, honest = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            idx = np.concatenate([rng.choice(lies, len(lies)), rng.choice(honest, len(honest))])
            gaps.append(faithfulness.auroc(y[idx], a[idx]) - faithfulness.auroc(y[idx], s[idx]))
        means[b] = np.mean(gaps)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def report(tag: str, scored: dict, substitute_name: str = "substitute through the target's classifier") -> dict:
    """Per-category gaps and their summary for one method, position, label and split; prints the summary line."""
    out = {"per_category": {}}
    for other in ("substitute", "own"):
        gaps = {c: faithfulness.gap(v["target"], v[other]) for c, v in scored.items() if other in v}
        if gaps:
            out["per_category"][other] = gaps
            out[other] = faithfulness.summary(gaps) | {"gap_ci": mean_gap_ci(scored, other)}
    if not out["per_category"]:   # no substitute: the target's own AUROC per category
        aurocs = {}
        for c, v in scored.items():
            y = np.array(v["target"]["label"])
            if min(int(y.sum()), int((y == 0).sum())) >= FEW:
                aurocs[c] = faithfulness.auroc(y, np.array(v["target"]["score"]))
        out["target_only"] = {"categories": len(aurocs), "target": float(np.mean(list(aurocs.values()))) if aurocs else None,
                              "per_category": aurocs}
        t = out["target_only"]
        print(f"gap | {tag}: AUROC {'-' if t['target'] is None else format(t['target'], '.3f')} "
              f"(mean over {t['categories']} categories)", flush=True)
        return out
    parts = []
    for other, name in (("substitute", substitute_name), ("own", "substitute with its own classifier")):
        if other in out and out[other]["categories"]:
            o = out[other]
            parts.append(f"{name} {o['substitute']:.3f}, gap {o['gap']:+.3f} [{o['gap_ci'][0]:+.3f}, {o['gap_ci'][1]:+.3f}] "
                         f"(clear gap in {o['gap_ci_excludes_zero']} of {o['categories']} categories)")
    first = next(o for o in (out.get("substitute"), out.get("own")) if o)
    print(f"gap | {tag}: target {'-' if first['target'] is None else format(first['target'], '.3f')} "
          f"(mean over {first['categories']} categories); " + "; ".join(parts), flush=True)
    return out


def top_words(rows: list[dict], feats: list, label: str) -> dict:
    """The words a classifier trained on every category (each weighted equally) leans on most, for and against a lie."""
    idx = [i for i, r in enumerate(rows) if r[label] is not None]
    counts = {c: sum(rows[i]["category"] == c for i in idx) for c in CATEGORIES}
    weight = np.array([1 / counts[rows[i]["category"]] for i in idx])
    vec = TfidfVectorizer(analyzer=lambda doc: doc, min_df=2, sublinear_tf=True)
    clf = faithfulness.words_classifier().fit(vec.fit_transform([feats[i] for i in idx]),
                                              np.array([rows[i][label] for i in idx]), sample_weight=weight)
    vocab, coef = np.array(vec.get_feature_names_out()), clf.coef_[0]
    order = np.argsort(coef)
    return {"lie": vocab[order[::-1][:TOP_WORDS]].tolist(), "truthful": vocab[order[:TOP_WORDS]].tolist()}


def load_pair() -> tuple[str, str, str, list[dict], dict, dict]:
    """--target and --substitute from the command line: (target, substitute, "target -> substitute", the rows both
    models read, the target's read of them, the substitute's read of them)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--substitute", required=True)
    args = ap.parse_args()
    pair = f"{args.target} -> {args.substitute}"
    rows = [json.loads(line) for line in open(LOCAL / f"rows_{args.target}.jsonl", encoding="utf-8")]
    t, s = load_read(args.target, args.target), load_read(args.target, args.substitute)
    rows = sorted((r for r in rows if r["uid"] in t and r["uid"] in s), key=lambda r: (r["category"], r["uid"]))
    print(f"rows | {pair}: {len(rows)} rows read by both models", flush=True)
    return args.target, args.substitute, pair, rows, t, s


def main() -> None:
    target, substitute, pair, rows, t, s = load_pair()
    result: dict = {"target": target, "substitute": substitute, "rows": len(rows), "gaps": {}, "top_words": {}}
    text = ["\n".join(f"{m['role']}: {m['content']}" for m in r["messages"][:-2]) for r in rows]   # the conversation before the follow-up
    for label_name, labels in LABELS.items():
        for split in SPLITS:
            tag = f"{pair} | words only | {label_name} | {split}"
            result["gaps"][tag] = report(tag, score(rows, text, None, "words only", labels, split))
            for position in POSITIONS:
                for method in (*WORD_METHODS, "probe"):
                    if not all(method in t[r["uid"]] and method in s[r["uid"]] for r in rows):
                        print(f"gap | {pair} | {method} | {position}: not computed, a model has no readout of this kind", flush=True)
                        continue
                    if method == "probe" and position != POSITIONS[0]:
                        continue   # the arena's probe reads before_answer (dyl_repro.core.probe.PROBE_POSITION)
                    ft, fs = ([m[r["uid"]][method][position] for r in rows] for m in (t, s))
                    tag = f"{pair} | {method} | {position} | {label_name} | {split}"
                    result["gaps"][tag] = report(tag, score(rows, ft, fs, method, labels, split))
    for method in WORD_METHODS:
        for name, model in (("target", t), ("substitute", s)):
            if all(method in model[r["uid"]] for r in rows):
                top = top_words(rows, [model[r["uid"]][method][POSITIONS[0]] for r in rows], "reply_lied")
                result["top_words"][f"{method} | {name}"] = top
                print(f"top words | {pair} | {method} | {name} | {POSITIONS[0]} | lie vs truthful: towards a lie "
                      f"{top['lie']}; towards truthful {top['truthful']}", flush=True)
    path = LOCAL / f"gap_{target}__{substitute}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
