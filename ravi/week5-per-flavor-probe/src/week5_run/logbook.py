"""Append-only run log. Each entry is flushed before the next experiment starts."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

HEADER = (
    "# Week 5 run log\n\n"
    "Appended by `run_week5.py`. Experiments run in order. "
    "A stage marked STOPPED was not followed by later stages.\n"
)


class Logbook:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(HEADER)

    def append(self, title: str, body: str = "") -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        text = f"\n## {stamp} — {title}\n\n{body.rstrip()}\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
