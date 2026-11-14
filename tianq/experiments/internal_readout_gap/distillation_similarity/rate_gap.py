"""Ours, not John's ask: can a readout tell where Qwen3.6 goes on to cheat, and does gemma's readout tell as well?

    python rate_gap.py          # prints every number it writes; writes OUT/distillation_similarity/rate_gap.json

At each hero swap spot Gigi measured how often Qwen3.6-27B does the task's bad act when it continues from there
(results/hero9_swap.csv, who = qwen36). The label here is "Qwen3.6 cheats from this spot at least half the time".
A method scores every spot with a classifier that never saw the spot's task (trained on the other tasks, each
weighted equally):
  J-lens words   TF-IDF over the J-lens top-200 words at 2/3 depth, logistic regression
  NLA words      TF-IDF over the NLA explanations' words, logistic regression
  activation     the activation at 2/3 depth, standardized, logistic regression (C = 0.01)
  words only     no model: TF-IDF over the history's own text (one- and two-word terms)
For Qwen3.6 itself ("target") and for base gemma and each distilled checkpoint as substitute: through the target's
classifier (the word methods; activations of two models don't line up) and through a classifier of its own.
    gap = target AUROC - substitute AUROC, both against Qwen3.6's label
The AUROC is taken inside each task and averaged over the tasks with at least MIN_PER_CLASS spots of each class.
Intervals: 95%, 2,000 paired resamples of spots inside each task.
Fixed before any result: the 0.5 cut, depth 2/3, leave one task out. Changed after the first result (13 Nov): the
first version pooled every spot into one AUROC. Every method then scored below 0.5, the words-only baseline included
(0.15), because Qwen3.6's label is nearly constant inside a task: each held-out task is scored by a classifier that
never saw it, and pooling those separately trained scores mostly measures which task a spot belongs to. The
within-task AUROC asks the intended question; the pooled number is still printed beside it.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import spots  # noqa: E402
from shared import models, readout, words  # noqa: E402
from similarity import EXPERIMENT, ORIGINAL, VERSIONS, lens_sets, load  # noqa: E402

CUT = 0.5
MIN_PER_CLASS = 5
WORD_METHODS = ("J-lens words", "NLA words")


def features(tag: str, names: list[str]) -> dict[str, list]:
    lens = load(tag, "lens")
    sets = lens_sets(lens, readout.HEADLINE)
    out = {"J-lens words": [sorted(sets[n]) for n in names]}
    first = lens[names[0]]
    layer = readout.layer_at(first["n_layers"], [int(k) for k in first["layers"]], readout.HEADLINE)
    out["activation"] = [torch.load(models.OUT / EXPERIMENT / tag / "acts" / f"{n}.pt")[0, layer].numpy() for n in names]
    nla = load(tag, "nla")
    if all(n in nla for n in names):
        out["NLA words"] = [words.text_words([s["explanation"] for s in nla[n]["positions"]["before_action"]]) for n in names]
    return out


def fit(method: str, x: list, y: np.ndarray, weight: np.ndarray):
    if method == "activation":
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000))
        model.fit(np.stack(x), y, logisticregression__sample_weight=weight)
        return lambda test: model.decision_function(np.stack(test))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True) if method == "words only" else TfidfVectorizer(
        analyzer=lambda doc: doc, min_df=2, sublinear_tf=True)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(vec.fit_transform(x), y, sample_weight=weight)
    return lambda test: clf.decision_function(vec.transform(test))


def held_out_scores(envs: list[str], y: np.ndarray, x_t: list, x_s: list | None, method: str) -> dict[str, np.ndarray]:
    s = {k: np.full(len(y), np.nan) for k in ("target", "substitute", "own")}
    for env in sorted(set(envs)):
        test = [i for i, e in enumerate(envs) if e == env]
        train = [i for i, e in enumerate(envs) if e != env]
        counts = {e: sum(envs[i] == e for i in train) for e in set(envs)}
        weight = np.array([1 / counts[envs[i]] for i in train])
        scorer = fit(method, [x_t[i] for i in train], y[train], weight)
        s["target"][test] = scorer([x_t[i] for i in test])
        if x_s is not None:
            if method in WORD_METHODS:
                s["substitute"][test] = scorer([x_s[i] for i in test])
            s["own"][test] = fit(method, [x_s[i] for i in train], y[train], weight)([x_s[i] for i in test])
    return s


def within_task(envs: list[str], y: np.ndarray, s: np.ndarray, idx: np.ndarray | None = None) -> float:
    """Mean over tasks of the AUROC inside the task, for tasks with at least MIN_PER_CLASS spots of each class.
    `idx` picks a resample of spots."""
    idx = np.arange(len(y)) if idx is None else idx
    aurocs = []
    for env in sorted(set(envs)):
        rows = np.array([i for i in idx if envs[i] == env], dtype=int)
        if len(rows) and min(y[rows].sum(), (1 - y[rows]).sum()) >= MIN_PER_CLASS:
            aurocs.append(roc_auc_score(y[rows], s[rows]))
    return float(np.mean(aurocs)) if aurocs else float("nan")


def gap(envs: list[str], y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """Target vs substitute by the within-task AUROC, with a 95% interval for the gap (spots resampled inside each
    task), and the all-spots-pooled AUROCs this analysis was first planned with."""
    rng = np.random.default_rng(words.SEED)
    by_env = {e: np.array([i for i, x in enumerate(envs) if x == e]) for e in set(envs)}
    diffs = []
    for _ in range(words.BOOT):
        idx = np.concatenate([rng.choice(rows, len(rows)) for rows in by_env.values()])
        diffs.append(within_task(envs, y, a, idx) - within_task(envs, y, b, idx))
    t, s = within_task(envs, y, a), within_task(envs, y, b)
    return {"target": t, "substitute": s, "gap": t - s, "ci": np.nanpercentile(diffs, [2.5, 97.5]).tolist(),
            "pooled_target": float(roc_auc_score(y, a)), "pooled_substitute": float(roc_auc_score(y, b))}


def main() -> None:
    beh = spots.behaviour()
    all_spots = {s.name: s for s in spots.spots()}
    read = set(load(ORIGINAL, "lens")).intersection(*(set(load(v, "lens")) for v in VERSIONS))
    names = sorted(n for n in read if beh.get(n, {}).get("qwen36", {}).get("scored"))
    envs = [all_spots[n].env for n in names]
    rate = np.array([beh[n]["qwen36"]["cheated"] / beh[n]["qwen36"]["scored"] for n in names])
    y = (rate >= CUT).astype(int)
    print(f"rate | {len(names)} spots read by every model with scored Qwen3.6 continuations, in {len(set(envs))} tasks; "
          f"Qwen3.6 cheats at least half the time at {int(y.sum())}", flush=True)
    result: dict = {"spots": len(names), "cheats": int(y.sum()), "methods": {}}
    by_point = spots.continuations()
    text = []
    for n in names:
        messages, _ = spots.history(all_spots[n], by_point[n])
        text.append("\n".join(f"{m['role']}: {m.get('content') or ''} {json.dumps(m.get('tool_calls') or '')}" for m in messages))
    scored = [e for e in sorted(set(envs)) if min(y[[i for i, x in enumerate(envs) if x == e]].sum(),
                                                   (1 - y[[i for i, x in enumerate(envs) if x == e]]).sum()) >= MIN_PER_CLASS]
    result["tasks_scored"] = scored
    print(f"rate | tasks with at least {MIN_PER_CLASS} spots of each class, where a within-task AUROC exists: {scored}", flush=True)
    s = held_out_scores(envs, y, text, None, "words only")
    result["methods"]["words only"] = {"target": within_task(envs, y, s["target"]),
                                       "pooled_target": float(roc_auc_score(y, s["target"]))}
    print(f"rate | words only (the history's text): within-task AUROC {result['methods']['words only']['target']:.3f} "
          f"(all spots pooled {result['methods']['words only']['pooled_target']:.3f})", flush=True)
    target = features(ORIGINAL, names)
    for v in VERSIONS:
        sub = features(v, names)
        for method in (*WORD_METHODS, "activation"):
            if method not in target or method not in sub:
                print(f"rate | {method} | {v}: not computed, a model has no readout of this kind", flush=True)
                continue
            s = held_out_scores(envs, y, target[method], sub[method], method)
            r = {k: gap(envs, y, s["target"], s[k]) for k in ("substitute", "own") if not np.isnan(s[k]).any()}
            result["methods"].setdefault(method, {})[v] = r
            parts = [f"{'through the target classifier' if k == 'substitute' else 'own classifier'} {g['substitute']:.3f}, "
                     f"gap {g['gap']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}] (pooled {g['pooled_substitute']:.3f})"
                     for k, g in r.items()]
            first = next(iter(r.values()))
            print(f"rate | {method} | within-task AUROC | {ORIGINAL} target {first['target']:.3f} (pooled "
                  f"{first['pooled_target']:.3f}); {v}: " + "; ".join(parts), flush=True)
    path = models.OUT / EXPERIMENT / "rate_gap.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
