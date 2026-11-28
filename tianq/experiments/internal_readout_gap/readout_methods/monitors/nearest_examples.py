"""A monitor with no classifier: a likeness method used as it is, against labelled example readouts.

A readout's score is how much more alike it is to the examples of class 1 than to those of class 0.
"""

from __future__ import annotations

import numpy as np

from ..likeness.pair import Pair
from ..readout_type import Readouts


def nearest_examples(method, train: Readouts, labels: np.ndarray, test: Readouts, **options) -> np.ndarray:
    """Per test readout: its mean likeness (by `method`, a function of a Pair in likeness/) to the training readouts
    labelled 1, minus its mean likeness to those labelled 0."""
    table = method(Pair(test, train, pool=[test, train]), **options)
    return table[:, labels == 1].mean(1) - table[:, labels == 0].mean(1)
