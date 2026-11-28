"""A monitor that is trained: readouts as rare-word vectors, then logistic regression.

A readout's vector is the one likeness/rare_words.py compares by cosine: pair.py's matrix (each word's worth, by
word_weights/value.py) times word_weights/rarity.py's weights. Two things differ, because this is a classifier:
the weights are counted over the training readouts only and applied unchanged to whatever is scored later, and a
word must be in at least MIN_LISTS training readouts to be a feature at all. Each vector is scaled to length 1.

`top` and `value` mean what they mean in rare_words.py. The lie-detection experiments use the arena's winners:
the top 5,000 words worth 1 each for J-lens readouts, every word worth its count for NLA sentences.

Until 27 Nov the weighting here was sklearn's TF-IDF, a second version of the same idea: ln((1 + n) / (1 + lists
with the word)) + 1 for rarity, and 1 + ln(count) for a repeated word. The numbers made with it and how far each
moved are in gender_secret_rerun/PLAN.md.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from ..likeness.pair import known, lists_with_word, matrix
from ..readout_type import Readouts
from ..word_weights import rarity

MIN_LISTS = 2


class RareWordFeatures:
    """Fitted on training readouts (`fit_transform`), then applied to any readouts (`transform`). `words` names
    the columns."""

    def __init__(self, top: int | None = None, value: str = "one") -> None:
        self.top, self.value = top, value

    def fit_transform(self, train: Readouts) -> sparse.csr_matrix:
        seen_at_all = {w: k for k, w in enumerate(sorted({w for ws, _ in train for w in ws[:self.top]}))}
        lists = lists_with_word(train, seen_at_all, self.top)
        self.words = [w for w, k in seen_at_all.items() if lists[k] >= MIN_LISTS]
        self.index = {w: k for k, w in enumerate(self.words)}
        self.weights = sparse.diags(rarity.weights(lists[lists >= MIN_LISTS], len(train)))
        return self.transform(train)

    def transform(self, readouts: Readouts) -> sparse.csr_matrix:
        return normalize(matrix(known(readouts, self.index, self.top), self.index, None, self.value) @ self.weights)


def train_word_classifier(train: Readouts, labels: np.ndarray, weight: np.ndarray | None = None,
                          top: int | None = None, value: str = "one") -> tuple[RareWordFeatures, LogisticRegression]:
    """The fitted features and classifier, for a caller that also wants the weight each word was given."""
    features = RareWordFeatures(top, value)
    return features, LogisticRegression(C=1.0, max_iter=2000).fit(features.fit_transform(train), labels, sample_weight=weight)


def word_classifier(train: Readouts, labels: np.ndarray, weight: np.ndarray | None = None,
                    top: int | None = None, value: str = "one"):
    """Train on readouts; returns a function scoring a list of readouts (higher: more like label 1)."""
    features, classifier = train_word_classifier(train, labels, weight, top, value)
    return lambda test: classifier.decision_function(features.transform(test))
