from __future__ import annotations

import sys
from pathlib import Path

WEEK3 = Path(__file__).resolve().parents[1]
WEEK1 = WEEK3.parent / "week1-dyl-vs-oa"
for path in (WEEK3 / "src", WEEK1 / "src", WEEK1 / "dyl-probe-repro"):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)
