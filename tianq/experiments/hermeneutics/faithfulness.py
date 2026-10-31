"""Model hermeneutics: the probe faithfulness gap (John, 2026-10-07; lesswrong.com/posts/CZLhfNGaDqtYxvGJD).

"Pretend like you have this target model and you have a substitute model ... pre-filled with the same
things, apply the exact same methodology ... The metric that we care the most about basically is
target minus substitute probe performance." (John, 7 Oct.)

The target (author) is a model reading its own conversations; a substitute (reader) is another model
prefilled with the target's exact conversations (core/model.py DYL_READER). For each category and
variant (diversity.VARIANTS), each model gets its own in-category probe, scored out of fold exactly as
the diversity chart's in-category reference (diversity.in_category): AISI's recipe at the model's own
probe layer, 5-fold, trained on the variant's training label and scored on its test label. Same rows,
same labels (the target's lies), and the same folds: rows sit in uid order in every model's cache
(store.ActivationCache), so StratifiedKFold(seed 0) splits them identically, which scores() checks.

    gap = target AUROC - substitute AUROC

with a 95% interval from a paired, class-stratified bootstrap over rows: each resample redraws the
lies and the truthful rows (with replacement, keeping their counts) and recomputes both AUROCs on
the same draw. A category counts in a mean when it has at least FEW of each scored class, as everywhere else.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from dyl_repro.core import training
from experiments.cross_category_transfer import diversity
from experiments.cross_category_transfer.generalization import CATEGORIES, FEW

BOOTSTRAP = 2000
SEED = 0
# The diversity chart's two variants. (A read before the reply is written was tried and dropped: the
# substitute sees only the prompt there, nothing the target wrote, so it says nothing about reading it.)
READS = tuple(diversity.VARIANTS)
# A substitute with no model at all: the conversation's words (TF-IDF, unigrams and bigrams, logistic
# regression), out of fold on the same rows and folds, for both reads. Under neutral both classes deny
# and the follow-up is the same in every row, so the conversation is the text that differs.
WORDS = "words only"


def scores(category: diversity.Category) -> dict | None:
    """The category's own probe, out of fold, on the rows its test label scores: uids, labels, scores.
    None when a training class is too small (diversity.in_category's rule)."""
    rows = category.trainable
    y = category.train_label[rows]
    if min(int((y == 1).sum()), int((y == 0).sum())) < training.MIN_ROWS_PER_CLASS:
        return None
    uids = [category.uids[i] for i in rows]
    if uids != sorted(uids):
        raise ValueError("rows are not in uid order: two models' folds would differ")
    s = training.out_of_fold_scores(category.x[rows], y, l2_lambda=training.DEFAULT_L2_LAMBDA)
    test = category.test_label[rows]
    kept = test != diversity.NOT_SCORED
    return {"uids": [u for u, k in zip(uids, kept) if k], "label": test[kept].astype(int).tolist(),
            "score": s[kept].tolist(), "trainable_uids": uids, "trainable_labels": y.astype(int).tolist()}


def words_vectorizer() -> TfidfVectorizer:
    """The words-only substitute's features. Single-character tokens kept: an ASCII reply is one letter."""
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")


def words_classifier() -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=2000)


def word_scores(target: dict, texts: dict[str, str]) -> dict:
    """The words-only substitute on the target's scored rows. Its folds are the probes': StratifiedKFold
    splits by label and row count only, and the probes split the same labels, in the same uid order,
    before their test-label filter; so the words are split on all trainable rows of the category too."""
    uids, labels = target["trainable_uids"], np.array(target["trainable_labels"])
    words = words_vectorizer()
    scores = np.empty(len(uids))
    for train, test in StratifiedKFold(training.CROSS_VALIDATION_FOLDS, shuffle=True, random_state=0).split(uids, labels):
        x = words.fit_transform([texts[uids[i]] for i in train])
        fit = words_classifier().fit(x, labels[train])
        scores[test] = fit.decision_function(words.transform([texts[uids[i]] for i in test]))
    by_uid = dict(zip(uids, scores))
    return {"uids": target["uids"], "label": target["label"], "score": [float(by_uid[u]) for u in target["uids"]]}


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC (ties count half), equal to sklearn's roc_auc_score."""
    ranks = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gap(target: dict, substitute: dict) -> dict:
    """Target vs substitute on one category's scored rows, matched by uid."""
    if target["uids"] != substitute["uids"] or target["label"] != substitute["label"]:
        raise ValueError("target and substitute were scored on different rows or labels")
    y = np.array(target["label"])
    a, b = np.array(target["score"]), np.array(substitute["score"])
    n_lie, n_honest = int(y.sum()), int((y == 0).sum())
    out = {"n_lie": n_lie, "n_honest": n_honest, "trusted": min(n_lie, n_honest) >= FEW}
    if not n_lie or not n_honest:
        return out | {"target": None, "substitute": None, "gap": None, "ci": None}
    lies, honest = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(SEED)
    diffs = np.empty(BOOTSTRAP)
    for i in range(BOOTSTRAP):
        idx = np.concatenate([rng.choice(lies, n_lie), rng.choice(honest, n_honest)])
        diffs[i] = auroc(y[idx], a[idx]) - auroc(y[idx], b[idx])
    t, s = auroc(y, a), auroc(y, b)
    return out | {"target": t, "substitute": s, "gap": t - s,
                  "ci": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]}


def summary(gaps: dict[str, dict]) -> dict:
    """Means over the categories with at least FEW of each class, and the gap's SD across them: the mean
    lets misses in opposite directions cancel, the SD shows how far single categories miss."""
    kept = [g for g in gaps.values() if g["trusted"] and g["gap"] is not None]
    mean = lambda k: float(np.mean([g[k] for g in kept])) if kept else None
    return {"categories": len(kept), "target": mean("target"), "substitute": mean("substitute"), "gap": mean("gap"),
            "gap_sd": float(np.std([g["gap"] for g in kept], ddof=1)) if len(kept) > 1 else None,
            "gap_ci_excludes_zero": sum(g["ci"][0] > 0 or g["ci"][1] < 0 for g in kept)}


def markdown(by_substitute: dict[str, dict[str, dict]], variant: str) -> str:
    """One variant: a row per category, target AUROC then each substitute's AUROC and gap [95% CI]."""
    names = list(by_substitute)
    lines = ["| category | lie / truthful | target | " + " | ".join(names) + " |",
             "|---" * (len(names) + 3) + "|"]
    first = by_substitute[names[0]][variant]
    for category in CATEGORIES:
        g0 = first.get(category)
        if g0 is None or g0["target"] is None:
            continue
        few = "" if g0["trusted"] else " (few)"
        cells = []
        for name in names:
            g = by_substitute[name][variant].get(category)
            cells.append("-" if g is None else
                         f"{g['substitute']:.2f} (gap {g['gap']:+.2f} [{g['ci'][0]:+.2f}, {g['ci'][1]:+.2f}])")
        lines.append(f"| {CATEGORIES[category]}{few} | {g0['n_lie']} / {g0['n_honest']} | {g0['target']:.2f} | "
                     + " | ".join(cells) + " |")
    means = [summary(by_substitute[n][variant]) for n in names]
    lines.append(f"| **mean ({means[0]['categories']} categories)** | | {means[0]['target']:.2f} | "
                 + " | ".join("-" if m["gap"] is None else f"{m['substitute']:.2f} (gap {m['gap']:+.3f})" for m in means)
                 + " |")
    return "\n".join(lines)
