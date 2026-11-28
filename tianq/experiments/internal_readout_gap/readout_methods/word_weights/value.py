"""What a word of a readout is worth before any other weighting.

  "p"              its value: the lens's probability, or the word's count in the NLA's sentences
  "one"            1 if the word is in the list at all
  "sqrt p"         the square root of its value (a softer confidence weight)
  "rank discount"  1 / log2(rank + 1): the first word 1, the third 0.5, the 1,000th about 0.1
"""

from __future__ import annotations

import numpy as np

KINDS = {
    "p": lambda values: values.tolist(),
    "one": lambda values: [1.0] * len(values),
    "sqrt p": lambda values: np.sqrt(values).tolist(),
    "rank discount": lambda values: (1 / np.log2(np.arange(2, len(values) + 2))).tolist(),
}


def worth(values: np.ndarray, kind: str) -> list[float]:
    """One number per word of a readout (its values, best first), by the chosen kind."""
    return KINDS[kind](values)
