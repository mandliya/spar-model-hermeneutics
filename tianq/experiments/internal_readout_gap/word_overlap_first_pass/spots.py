# ARCHIVED (8 Nov 2026 first pass). Kept only so that experiment can be reproduced; do not import or extend.
# For new work use ../shared/ (takeover spots for the hero swap).
"""Gigi's takeover spots: the exact history each model saw there, and the behaviour measured there.

A spot is a saved target run rewound to step K (gigi/3-forensics/results/swap_*_plan.json, read through his
swap_all_score.PLAN). His harness resumes from `<run>/step-K/messages.json` (agent-interp-envs/scripts/resume.py), so
that file is the history; the next turn a model writes is turn K+1. Tools come from the environment's own tools.py
for the run's config (his distill_data.tools_for). Behaviour per spot and model comes from his results/gap_replays.csv
(who = qwen: the target continuing itself; who = gemma: the substitute), one row per try.

Nothing here is reimplemented: Gigi's modules are imported as they are, with his paths.
"""

from __future__ import annotations

import collections
import csv
import json
import pathlib
import sys
from dataclasses import dataclass

REPO = pathlib.Path(__file__).resolve().parents[4]
FORENSICS = REPO / "gigi" / "3-forensics"
sys.path.insert(0, str(FORENSICS / "scripts"))
import distill_data as D  # noqa: E402  (also puts agent-interp-envs on the path, imports swap_all_score as S)

S = D.S


@dataclass(frozen=True)
class Spot:
    name: str          # Gigi's point name (swap_all_score.point_name)
    env: str
    run: str           # the target run, relative to gigi/3-forensics
    K: int             # takeover step: the history ends after turn K
    N: int             # the turn Gigi's plan marks (the cheat for incident spots)
    incident: bool     # True: rewound from a run where the target cheated


def spots() -> list[Spot]:
    """Every spot in Gigi's swap plan, in his order."""
    return [Spot(S.point_name(p), p["env"], p["run"], int(p["K"]), int(p["N"]), bool(p["incident"])) for p in S.PLAN]


def behaviour() -> dict[str, dict[str, dict[str, int]]]:
    """point -> who -> {"cheated", "scored", "tries"}: tries counts every row, scored those with a verdict."""
    out: dict = collections.defaultdict(lambda: collections.defaultdict(lambda: {"cheated": 0, "scored": 0, "tries": 0}))
    for row in csv.DictReader(open(FORENSICS / "results" / "gap_replays.csv", encoding="utf-8")):
        cell = out[row["point"]][row["who"]]
        cell["tries"] += 1
        if row["hacked"] != "":
            cell["scored"] += 1
            cell["cheated"] += int(row["hacked"])
    return {p: dict(w) for p, w in out.items()}


def history(spot: Spot) -> tuple[list[dict], list[dict]]:
    """(messages, tools) exactly as the harness resumes at this spot: OpenAI-schema messages from step-K."""
    run = FORENSICS / spot.run
    messages = json.loads((run / f"step-{spot.K}" / "messages.json").read_text())
    config = D.yaml.safe_load((run.parent / "config.yaml").read_text())
    return messages, D.tools_for(config)


def gemma_messages(messages: list[dict], tools: list[dict]) -> list[dict]:
    """The same history as gemma receives it in text mode (Gigi's harness renderer, byte-identical to inference)."""
    return D._text_tools_messages(messages, tools)
