from __future__ import annotations

import sys
from pathlib import Path

WEEK6 = Path(__file__).resolve().parents[1]
WEEK3 = WEEK6.parent / "week3-noninvasive-probe"
WEEK1 = WEEK6.parent / "week1-dyl-vs-oa"
for path in (WEEK6 / "src", WEEK3 / "src", WEEK1 / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
