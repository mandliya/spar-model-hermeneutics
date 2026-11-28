"""Loading for the arena's J-space runs: which models and lenses, cut down to the restart points all of them have.

The loading itself (a read's top 5,000 words with probabilities, at every depth and restart point) is
readout_methods/word_lists/jlens_lists.py; its input files are in the folder READOUT_DISTRIBUTIONS.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # internal_readout_gap/: readout_methods
from readout_methods.word_lists.jlens_lists import FOLDER, Read  # noqa: E402,F401  (FOLDER: where the runs write their results)

QWEN, BASE, FINAL = "qwen3.6-27b", "gemma-3-27b-it", "gemma-distilled-step96"
# which lens each gemma is read through; Qwen3.6 is always read through its published lens
LENSES = {"borrowed lens": {BASE: "published", FINAL: "published"}, "own lens": {BASE: "control-lens", FINAL: "final-lens"}}


def load_shared(keys: list[tuple[str, str]]) -> tuple[dict[tuple[str, str], Read], list[str]]:
    """The (model, lens) reads asked for, cut down to the restart points every one of them has, in one order."""
    reads = {key: Read(*key) for key in keys}
    names = sorted(set.intersection(*(set(r.names) for r in reads.values())))
    for key, r in reads.items():
        print(f"read | {key[0]} through its {key[1]} lens: {len(r.names)} restart points, {len(names)} shared by every read", flush=True)
        r.select(names)
    return reads, names
