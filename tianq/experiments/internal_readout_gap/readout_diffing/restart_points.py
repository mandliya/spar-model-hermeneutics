"""Which task each restart point belongs to (Gigi's plan, read through distillation_similarity/spots.py)."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "distillation_similarity"))
import spots  # noqa: E402


def tasks(names: list[str]) -> list[str]:
    """Each restart point's task, in the order of `names`."""
    task = {s.name: s.env for s in spots.spots()}
    return [task[n] for n in names]
