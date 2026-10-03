"""Everything this dataset takes from Gigi's ascii-sandbagging project.

The only module that names Gigi's ASCII files. His src/common.py is loaded
through the gigi package's module() (its load_config and to_messages define how his
dataset is read); the follow-up questions and their parsers are in the shared
master list, ../followups.json, verified against the same file:
https://github.com/mandliya/spar-model-hermeneutics/blob/268c51c/gigi/ascii-sandbagging/src/common.py

Two things live inside Gigi's scripts rather than importable functions (the
scripts run on import), so they are restated here, each with its permalink: the
dataset config name, and the conversation order. Gigi's recorded answers are
read straight from their results/ CSVs.
"""

from __future__ import annotations

import re
from functools import cache

from dyl_repro.datasets import gigi

# The checkout these CSVs are read at (2026-09-27). Rows extracted earlier record
# the commit of their own run (fa64456) in their provenance.
GIGI_COMMIT = gigi.GIGI_COMMIT
CATEGORY = "ascii-sandbagging"
_common = gigi.module(CATEGORY, "common")

# Restated from 00_ascii_activations.py L62:
# https://github.com/mandliya/spar-model-hermeneutics/blob/fa64456/gigi/ascii-sandbagging/00_ascii_activations.py#L62
CONFIG = "google-gemma-3-27b-it_task-group_sandbagging"

# Gigi's condition -> the CSV holding his recorded answers. `neutral` lives in
# ascii_results.csv as `followup_answer`; the rest in his condition tables. He
# generated them greedily (temperature 0): his determinism check regenerated 50
# saved answers exactly (gigi/unanswerable/results/temp_determinism_check.json).
RECORDED_CSV = {
    "neutral": "ascii_results.csv",
    "low_stakes": "ascii_followup_conditions.csv",
    "high_stakes": "ascii_followup_conditions.csv",
    "confidence": "ascii_followup_conditions.csv",
    "plain": "ascii_benchmark_conditions.csv",
    "named": "ascii_benchmark_conditions.csv",
    "eval_plain": "ascii_benchmark_conditions.csv",
    "eval_named": "ascii_benchmark_conditions.csv",
}

@cache
def dataset_rows() -> dict[str, dict]:
    """The ASCII rows of Gigi's config, by sample_id, loaded exactly as
    00_ascii_activations.py L142-144 loads them."""
    import pandas as pd

    df = pd.concat([_common.load_config(CONFIG, "train"), _common.load_config(CONFIG, "validation")])
    df = df[df["task"].str.contains("ascii", case=False, na=False)]
    return {row["sample_id"]: row for _, row in df.iterrows()}


@cache
def ascii_results() -> dict[str, dict]:
    """Gigi's 310 ASCII rows: their `lied` label and Gemma's first answer."""
    return {r["uid"]: r for r in gigi.csv_rows(CATEGORY, "ascii_results.csv")}


def lied(uid: str) -> bool | None:
    return gigi.BOOL[ascii_results()[uid]["lied"]]


def conversation_before_followup(uid: str) -> list[dict[str, str]]:
    """trace + capability question + Gemma's own answer, as Gigi builds it in
    01_dyl_recapture_ascii.py L104-110 (build_convo), minus the follow-up:
    https://github.com/mandliya/spar-model-hermeneutics/blob/fa64456/gigi/ascii-sandbagging/01_dyl_recapture_ascii.py#L104-L110
    """
    row = dataset_rows()[uid]
    probe = dict(row["lie_detection_prompt"])
    messages = _common.to_messages(row["trace"]) + [
        {"role": str(probe["role"]), "content": str(probe["content"])},
        {"role": "assistant", "content": str(ascii_results()[uid]["probe_answer"])},
    ]
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]


@cache
def recorded_answers(condition: str | None) -> dict[str, dict]:
    """uid -> {"answer", "gigi_admitted"} for one of Gigi's conditions; {} for None (never asked by him)."""
    if condition is None:
        return {}
    if condition == "neutral":
        return {r["uid"]: {"answer": r["followup_answer"], "gigi_admitted": gigi.BOOL[r["admitted"]]}
                for r in gigi.csv_rows(CATEGORY, RECORDED_CSV[condition])}
    return gigi.recorded_answers(CATEGORY, RECORDED_CSV[condition], condition)


def topic(uid: str) -> str:
    """The ASCII subject category encoded in the uid, e.g. ..._ascii_mc_medical_5 -> medical."""
    match = re.search(r"_ascii_mc_(.+)_\d+$", uid)
    if match is None:
        raise ValueError(f"no topic in uid {uid!r}")
    return match.group(1)
