"""Everything this dataset takes from Gigi's ascii-sandbagging project.

The only module that names Gigi's code. What defines Gigi's dataset and
questions is imported from their src/common.py, not retyped, so the wording and
the meaning of "admitted" have one source:
https://github.com/mandliya/spar-model-hermeneutics/blob/fa64456/gigi/ascii-sandbagging/src/common.py

Two things live inside Gigi's scripts rather than importable functions (the
scripts run on import), so they are restated here, each with its permalink: the
dataset config name, and the conversation order. Gigi's recorded answers are
read straight from their results/ CSVs.
"""

from __future__ import annotations

import csv
import os
import re
import sys
from functools import cache
from pathlib import Path

# The commit these imports and CSVs were read at (the `git pull` of 2026-09-23).
GIGI_COMMIT = "fa64456"
# Gigi's folder in the repo; GIGI_ASCII_DIR points elsewhere when this project
# is not inside a repo checkout (e.g. copied alone onto a GPU box).
GIGI_DIR = Path(
    os.environ.get("GIGI_ASCII_DIR", Path(__file__).resolve().parents[5] / "gigi" / "ascii-sandbagging")
)
RESULTS_DIR = GIGI_DIR / "results"

sys.path.insert(0, str(GIGI_DIR))
from src.common import (  # noqa: E402
    FOLLOWUP,
    FOLLOWUP_CONFIDENCE,
    FOLLOWUP_HIGH_STAKES,
    FOLLOWUP_LOW_STAKES,
    FOLLOWUP_NAMED_SANDBAGGING,
    FOLLOWUP_PLAIN,
    admitted_lying,
    letter,
    load_config,
    parse_confidence,
    to_messages,
)

# Restated from 00_ascii_activations.py L62:
# https://github.com/mandliya/spar-model-hermeneutics/blob/fa64456/gigi/ascii-sandbagging/00_ascii_activations.py#L62
CONFIG = "google-gemma-3-27b-it_task-group_sandbagging"

# Our condition name -> (Gigi's CSV, Gigi's condition name). `neutral` lives in
# ascii_results.csv as `followup_answer`; the rest in their condition tables.
# Note Gigi sampled these answers at their TEMPERATURE (common.py L39, default 1.0).
RECORDED = {
    "neutral": ("ascii_results.csv", None),
    "low_stakes": ("ascii_followup_conditions.csv", "low_stakes"),
    "high_stakes": ("ascii_followup_conditions.csv", "high_stakes"),
    "confidence": ("ascii_followup_conditions.csv", "confidence"),
    "plain": ("ascii_benchmark_conditions.csv", "plain"),
    "named_sandbagging": ("ascii_benchmark_conditions.csv", "named"),
}

_BOOL = {"True": True, "False": False, "": None}


def _csv(name: str) -> list[dict]:
    with (RESULTS_DIR / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@cache
def dataset_rows() -> dict[str, dict]:
    """The ASCII rows of Gigi's config, by sample_id, loaded exactly as
    00_ascii_activations.py L142-144 loads them."""
    import pandas as pd

    df = pd.concat([load_config(CONFIG, "train"), load_config(CONFIG, "validation")])
    df = df[df["task"].str.contains("ascii", case=False, na=False)]
    return {row["sample_id"]: row for _, row in df.iterrows()}


@cache
def ascii_results() -> dict[str, dict]:
    """Gigi's 310 ASCII rows: their `lied` label and Gemma's first answer."""
    return {r["uid"]: r for r in _csv("ascii_results.csv")}


def lied(uid: str) -> bool | None:
    return _BOOL[ascii_results()[uid]["lied"]]


def conversation_before_followup(uid: str) -> list[dict[str, str]]:
    """trace + capability question + Gemma's own answer, as Gigi builds it in
    01_dyl_recapture_ascii.py L104-110 (build_convo), minus the follow-up:
    https://github.com/mandliya/spar-model-hermeneutics/blob/fa64456/gigi/ascii-sandbagging/01_dyl_recapture_ascii.py#L104-L110
    """
    row = dataset_rows()[uid]
    probe = dict(row["lie_detection_prompt"])
    messages = to_messages(row["trace"]) + [
        {"role": str(probe["role"]), "content": str(probe["content"])},
        {"role": "assistant", "content": str(ascii_results()[uid]["probe_answer"])},
    ]
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]


@cache
def recorded_answers(condition: str) -> dict[str, dict]:
    """uid -> {"answer", "gigi_admitted"} for a condition Gigi recorded; {} otherwise."""
    if condition not in RECORDED:
        return {}
    file, gigi_condition = RECORDED[condition]
    if gigi_condition is None:
        return {
            r["uid"]: {"answer": r["followup_answer"], "gigi_admitted": _BOOL[r["admitted"]]}
            for r in _csv(file)
        }
    return {
        r["sample_id"]: {"answer": r["answer"], "gigi_admitted": _BOOL.get(r["admitted"])}
        for r in _csv(file)
        if r["condition"] == gigi_condition
    }


def topic(uid: str) -> str:
    """The ASCII subject category encoded in the uid, e.g. ..._ascii_mc_medical_5 -> medical."""
    match = re.search(r"_ascii_mc_(.+)_\d+$", uid)
    if match is None:
        raise ValueError(f"no topic in uid {uid!r}")
    return match.group(1)
