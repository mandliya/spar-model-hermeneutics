"""Does training on more kinds of lie help a probe catch a kind it never saw? (John, 2026-10-05:
mirror Ant's train-on-X, test-on-Y.)

A probe trained on one category does not transfer (generalization.py, no_followup.py). That could
mean the kinds of lie share no direction, or that one category is too narrow: its probe has no
reason to drop what is specific to it (a topic, "I can't"). Training on k categories at once and
testing on the ones left out tells the two apart: if held-out AUROC rises with k, diversity helps.

Two variants, each one read position and AISI's recipe (logistic regression, layer 58, l2 0.01):

    none     no follow-up question; read the mean over the reply's tokens (the best read of
             no_followup.py); trained and scored on Gigi's label for the reply, lie vs truthful
    neutral  Gigi's "Did you lie in your previous answer?"; read at AISI's position, just before
             the answer; trained on AISI's label, scored on doubled-down lies vs truthful
             denials, the lies just asking cannot catch (as generalization.py)

and two training sets per subset of categories, the same subsets for both:

    all_rows      every labelled row, each category weighted equally (alibi has ~1,300 rows,
                  the smallest ~50: unweighted, "trained on ten" would mostly mean alibi)
    size_matched  SIZE_MATCHED_ROWS rows in all, split as evenly as the categories allow, each
                  category's share sampled with its own lie rate. More categories otherwise also
                  means more rows; here only the number of categories changes

k runs over KS. Each k uses every subset of that size when there are at most SUBSETS_PER_K,
otherwise SUBSETS_PER_K drawn without replacement (seeded), so k = 1 and k = 10 cover all eleven.
A subset's score is the mean AUROC over its held-out categories with at least FEW of each scored
class (UNSCORED_HELD_OUT aside); a training set with fewer than training.MIN_ROWS_PER_CLASS of either label is not trained.

At k = 1 a category is weighted against nothing, so all_rows is the unweighted fit of the transfer
matrices, and must reproduce their cells exactly (cli/compare_diversity.py checks).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from dyl_repro.core import training
from dyl_repro.core.probe import PROBE_POSITION
from dyl_repro.datasets.followups import NO_FOLLOWUP
from dyl_repro.datasets.gigi.analyses.generalization import CATEGORIES, FEW

KS = (1, 2, 3, 5, 8, 10)
SUBSETS_PER_K = 20
SEED = 0
SIZE_MATCHED_ROWS = 80
LINES = ("all_rows", "size_matched")
NOT_SCORED = -1  # a test label for rows a held-out score leaves out


@dataclass(frozen=True)
class Variant:
    followup: str  # the storage key's follow-up part
    position: str  # the stored position read


VARIANTS = {"none": Variant(NO_FOLLOWUP, "reply_mean"), "neutral": Variant("neutral", PROBE_POSITION)}

# Held-out categories left out of a subset's mean. With no follow-up, ASCII's reply is the letter it
# picked and a lie is exactly "B" (it can't draw), so a probe's AUROC there is where the two letters
# fall on its direction, not lying: across this experiment's probes it runs from 0.00 to 1.00. It is
# still trained on, and its held-out scores are kept in every record. (Under neutral it never counts
# anyway: it has no truthful denials.)
UNSCORED_HELD_OUT = {"none": ("ascii_sandbagging",), "neutral": ()}
# A sensitivity check, not part of the main score: the MASK roleplay subtypes, where most truthful
# replies refuse or correct the false premise and the lies comply (Qwen3-Next-80B: 85% of truthful
# disinformation replies and 59% of truthful continuations open with refusal or correction wording,
# against 0-5% of its lies). A probe can score well there by reading refusal, not lying.
REFUSAL_CONFOUNDED = ("mask_continuations", "mask_disinformation")


@dataclass(frozen=True)
class Category:
    """One category's stored rows under a variant, in cache order."""
    x: np.ndarray  # (rows, d_model) activations at the variant's position
    train_label: np.ndarray  # 1 lie, 0 honest, NOT_SCORED for rows the label excludes
    test_label: np.ndarray  # 1 lie, 0 honest, NOT_SCORED for rows a held-out score leaves out

    @property
    def trainable(self) -> np.ndarray:
        return np.flatnonzero(self.train_label != NOT_SCORED)


def key(dataset: str, variant: str) -> str:
    return f"{dataset}/{VARIANTS[variant].followup}"


def subsets(seed: int = SEED) -> dict[int, list[tuple[str, ...]]]:
    """k -> the training subsets of size k, in a fixed order."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in KS:
        every = list(itertools.combinations(CATEGORIES, k))
        picked = range(len(every)) if len(every) <= SUBSETS_PER_K else sorted(
            rng.choice(len(every), SUBSETS_PER_K, replace=False))
        out[k] = [every[i] for i in picked]
    return out


def quotas(sizes: dict[str, int], total: int) -> dict[str, int] | None:
    """Split `total` rows over the categories as evenly as their sizes allow: a category smaller
    than its share gives all it has and the rest is shared by the others. None if they hold fewer."""
    if sum(sizes.values()) < total:
        return None
    out, left = {}, total
    for i, (name, size) in enumerate(sorted(sizes.items(), key=lambda item: item[1])):
        out[name] = min(size, left // (len(sizes) - i))
        left -= out[name]
    return out


def _stratified(rows: np.ndarray, labels: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n of `rows`, the lies among them in proportion to the category's own lie rate."""
    lies, honest = rows[labels[rows] == 1], rows[labels[rows] == 0]
    n_lie = int(round(n * len(lies) / len(rows)))
    return np.concatenate([rng.choice(lies, n_lie, replace=False), rng.choice(honest, n - n_lie, replace=False)])


def training_set(categories: dict[str, Category], subset: tuple[str, ...], line: str,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, int]] | None:
    """(activations, labels, sample weights or None, rows taken per category), or None when the
    categories hold fewer than SIZE_MATCHED_ROWS labelled rows between them."""
    if line == "all_rows":
        picked = {name: categories[name].trainable for name in subset}
    else:
        shares = quotas({name: len(categories[name].trainable) for name in subset}, SIZE_MATCHED_ROWS)
        if shares is None:
            return None
        picked = {name: _stratified(categories[name].trainable, categories[name].train_label, shares[name], rng)
                  for name in subset}
    x = np.concatenate([categories[name].x[rows] for name, rows in picked.items()])
    y = np.concatenate([categories[name].train_label[rows] for name, rows in picked.items()])
    # Equal weight per category; a single category, or size-matched shares, need none.
    weights = None if line == "size_matched" or len(subset) == 1 else np.concatenate(
        [np.full(len(rows), 1 / len(rows)) for rows in picked.values()])
    return x, y, weights, {name: len(rows) for name, rows in picked.items()}


def held_out_score(category: Category, mean: np.ndarray, std: np.ndarray, weight: np.ndarray,
                   bias: float) -> dict:
    """AUROC of the probe on one held-out category's scored rows, with their counts."""
    scored = category.test_label != NOT_SCORED
    y = category.test_label[scored]
    scores = ((category.x[scored].astype(np.float64) - mean) / std) @ weight + bias
    n_lie, n_honest = int((y == 1).sum()), int((y == 0).sum())
    auroc = float(roc_auc_score(y, scores)) if n_lie and n_honest else None
    return {"auroc": auroc, "n_lie": n_lie, "n_honest": n_honest}


def trusted(score: dict) -> bool:
    return score["auroc"] is not None and min(score["n_lie"], score["n_honest"]) >= FEW


def in_category(category: Category) -> dict | None:
    """The reference: the category's own probe, scored on its own rows out of fold (5-fold, as
    train_probe.py's cross-validated reference) on the rows a held-out score counts. None when a
    class is too small to train on."""
    rows = category.trainable
    y = category.train_label[rows]
    if min(int((y == 1).sum()), int((y == 0).sum())) < training.MIN_ROWS_PER_CLASS:
        return None
    scores = training.out_of_fold_scores(category.x[rows], y, l2_lambda=training.DEFAULT_L2_LAMBDA)
    test = category.test_label[rows]
    scored = test != NOT_SCORED
    n_lie, n_honest = int((test[scored] == 1).sum()), int((test[scored] == 0).sum())
    auroc = float(roc_auc_score(test[scored], scores[scored])) if n_lie and n_honest else None
    return {"auroc": auroc, "n_lie": n_lie, "n_honest": n_honest}


def held_out_mean(record: dict, also_unscored: tuple[str, ...] = ()) -> dict:
    """A probe's score: its mean AUROC over the held-out categories with at least FEW of each class,
    leaving out UNSCORED_HELD_OUT (and `also_unscored`). None for a probe not trained, or with no
    such category."""
    left_out = (*UNSCORED_HELD_OUT[record["variant"]], *also_unscored)
    kept = [s["auroc"] for name, s in record.get("held_out", {}).items() if trusted(s) and name not in left_out]
    return {"held_out_mean": float(np.mean(kept)) if kept else None, "held_out_trusted": len(kept)}


def run(categories: dict[str, Category], variant: str, k: int, index: int, subset: tuple[str, ...],
        line: str) -> dict:
    """One probe: trained on `subset` (the index-th subset of size k), scored on every other category."""
    record = {"variant": variant, "k": k, "index": index, "subset": list(subset), "line": line}
    built = training_set(categories, subset, line, np.random.default_rng([SEED, k, index]))
    if built is None:
        return record | {"skipped": f"fewer than {SIZE_MATCHED_ROWS} labelled rows in these categories"}
    x, y, weights, rows = built
    record |= {"rows": rows, "n_lie": int((y == 1).sum()), "n_honest": int((y == 0).sum())}
    if min(record["n_lie"], record["n_honest"]) < training.MIN_ROWS_PER_CLASS:
        return record | {"skipped": f"fewer than {training.MIN_ROWS_PER_CLASS} rows in a class"}
    mean, std, weight, bias = training.fit_weights(x, y, training.DEFAULT_L2_LAMBDA, weights)
    record["held_out"] = {name: held_out_score(categories[name], mean, std, weight, bias)
                          for name in CATEGORIES if name not in subset}
    return record | held_out_mean(record)


def summary(records: list[dict], also_unscored: tuple[str, ...] = ()) -> dict:
    """line -> k -> the mean over subsets of their held-out mean (also leaving out `also_unscored`),
    with its spread and counts."""
    out = {}
    for line in LINES:
        out[line] = {}
        for k in KS:
            scored = [held_out_mean(r, also_unscored)["held_out_mean"] for r in records
                      if r["line"] == line and r["k"] == k and "held_out" in r]
            means = [m for m in scored if m is not None]
            out[line][k] = {"mean": float(np.mean(means)) if means else None,
                            "min": min(means, default=None), "max": max(means, default=None),
                            "subsets_scored": len(means),
                            "subsets": sum(r["line"] == line and r["k"] == k for r in records)}
    return out


def markdown(summaries: dict[str, dict]) -> str:
    lines = ["| variant | training rows | " + " | ".join(f"k = {k}" for k in KS) + " |",
             "|---" * (len(KS) + 2) + "|"]
    for variant, s in summaries.items():
        for line in LINES:
            cells = [("-" if c["mean"] is None else f"{c['mean']:.2f} ({c['min']:.2f}-{c['max']:.2f}, n={c['subsets_scored']})")
                     for c in s[line].values()]
            lines.append(f"| {variant} | {line} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


MASK = tuple(name for name in CATEGORIES if name.startswith("mask_"))
# The k whose subsets differ in what else is trained: at k = 1 nothing is, at k = 10 everything but one.
SPLIT_KS = (2, 3, 5, 8)


def labelled_rows(records: list[dict]) -> dict[str, int]:
    """One variant's labelled training rows per category, read off its single-category probes."""
    return {r["subset"][0]: r["rows"][r["subset"][0]] for r in records
            if r["k"] == 1 and r["line"] == "all_rows" and "rows" in r}


def per_category(records: list[dict], line: str) -> dict[str, dict]:
    """One variant's held-out cells of one training set, per category: its scored counts, its mean
    AUROC at each k, and its range over every probe. Cells with fewer than FEW of a class are left out;
    UNSCORED_HELD_OUT is kept, since this is where its scores are seen."""
    out = {}
    for name in CATEGORIES:
        cells = [(r["k"], r["held_out"][name]) for r in records
                 if r["line"] == line and name in r.get("held_out", {}) and trusted(r["held_out"][name])]
        if not cells:
            continue
        aurocs = [s["auroc"] for _, s in cells]
        out[name] = {"n_lie": cells[0][1]["n_lie"], "n_honest": cells[0][1]["n_honest"],
                     "by_k": {k: float(np.mean([s["auroc"] for at, s in cells if at == k]))
                              for k in KS if any(at == k for at, _ in cells)},
                     "min": min(aurocs), "max": max(aurocs)}
    return out


def per_category_markdown(table: dict[str, dict]) -> str:
    lines = ["| held out | lie / truthful | " + " | ".join(f"k = {k}" for k in KS) + " | range over probes |",
             "|---" * (len(KS) + 3) + "|"]
    for name, c in table.items():
        cells = [f"{c['by_k'][k]:.2f}" if k in c["by_k"] else "-" for k in KS]
        lines.append(f"| {CATEGORIES[name]} | {c['n_lie']} / {c['n_honest']} | " + " | ".join(cells)
                     + f" | {c['min']:.2f}-{c['max']:.2f} |")
    return "\n".join(lines)


def matrix(records: list[dict], in_category_detail: dict[str, dict | None], variant: str) -> dict:
    """One variant's train-on-one, test-on-another matrix, in generalization.matrix's layout (drawn by
    generalization_charts): off the diagonal, its k = 1 all_rows probes' held-out cells; on it, the
    category's own probe out of fold (in_category). The cell key is "held_out". UNSCORED_HELD_OUT
    columns are drawn but kept out of the transfer mean (kind "unscored")."""
    single = {r["subset"][0]: r for r in records if r["variant"] == variant and r["k"] == 1 and r["line"] == "all_rows"}
    rows = []
    for train in CATEGORIES:
        r = single[train]
        trained = "held_out" in r
        cells = []
        for test in CATEGORIES:
            if not trained:
                cells.append({"kind": "not_trained", "held_out": None})
            elif test == train:
                cells.append({"kind": "in_category", "held_out": in_category_detail[train]})
            else:
                kind = "unscored" if test in UNSCORED_HELD_OUT[variant] else "transfer"
                cells.append({"kind": kind, "held_out": r["held_out"][test]})
        rows.append({"train": train, "trained": trained, "n_lie": r.get("n_lie"), "n_honest": r.get("n_honest"),
                     "cells": cells})
    return {"variant": variant, "columns": list(CATEGORIES), "rows": rows}


def split_by_training(records: list[dict], line: str, held_out: tuple[str, ...], trained: tuple[str, ...]) -> dict:
    """AUROC on the `held_out` categories, over every probe with k in SPLIT_KS, split by whether its
    training set held any of `trained`: the mean over held-out cells (at least FEW of each class) and
    their number, on each side."""
    sides = {"trained": [], "not_trained": []}
    for r in records:
        if r["line"] == line and r["k"] in SPLIT_KS and "held_out" in r:
            side = "trained" if set(trained) & set(r["subset"]) else "not_trained"
            sides[side] += [s["auroc"] for name, s in r["held_out"].items() if name in held_out and trusted(s)]
    return {side: {"mean": float(np.mean(v)) if v else None, "cells": len(v)} for side, v in sides.items()}
