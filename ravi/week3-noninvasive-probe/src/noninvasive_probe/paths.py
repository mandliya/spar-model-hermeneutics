"""Week 3 lives next to week 1; we reuse week 1 loaders and Gemma extract helpers."""

from __future__ import annotations

import sys
from pathlib import Path

# paths.py lives in week3/src/noninvasive_probe/
WEEK3 = Path(__file__).resolve().parents[2]
WEEK1 = WEEK3.parent / "week1-dyl-vs-oa"


def bootstrap() -> tuple[Path, Path]:
    for path in (WEEK3 / "src", WEEK1 / "src", WEEK1 / "dyl-probe-repro"):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
    return WEEK3, WEEK1
