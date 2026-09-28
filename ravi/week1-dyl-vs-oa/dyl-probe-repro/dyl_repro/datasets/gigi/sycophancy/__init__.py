"""Gigi's sycophancy rows, both tracks: does the official DYL probe transfer?

    answer.py    Track A: Gemma's trivia answer folds to the belief the user states
    feedback.py  Track B: Gemma's critique of an argument shifts with a social cue

Each conversation is the one Gigi's follow-up run sent, build_convo in
02_predecision_sycophancy_battery.py L117-127: the dataset trace, its
lie_detection_prompt (the pressured ask), Gemma's saved reply to it, then the
follow-up:
https://github.com/mandliya/spar-model-hermeneutics/blob/268c51c/gigi/sycophancy/02_predecision_sycophancy_battery.py#L117-L127
"""

from __future__ import annotations

from dyl_repro.datasets import gigi

CATEGORY = "sycophancy"
CONFIG = "google-gemma-3-27b-it_task-group_sycophancy"


def prefixes(task_prefix: str, replies: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    """uid -> trace + pressured ask + Gemma's saved reply."""
    rows = gigi.dataset_rows(CATEGORY, CONFIG, task_prefix, drop_duplicate_ids=False)
    return {
        uid: gigi.trace_messages(CATEGORY, rows[uid]) + [{"role": "assistant", "content": reply}]
        for uid, reply in replies.items()
    }
