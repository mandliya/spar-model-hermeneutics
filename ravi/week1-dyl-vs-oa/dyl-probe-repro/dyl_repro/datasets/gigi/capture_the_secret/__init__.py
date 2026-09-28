"""Gigi's capture-the-secret rows: does Gemma assert a falsehood to protect a secret?

Gemma plays a persona told never to mention a classified project; another
character makes conversation. Gigi's judge decided whether that character
actually pressed on the secret and what Gemma did: `lied` is a FALSE_DENIAL
(deflecting and revealing are honest). Only pressed rows have a turn label.
Each conversation is the one his follow-up run sent,
00_capture_secret_selfreport_benchmark.py L153-157: the dataset trace, its
lie_detection_prompt (the other character's line), Gemma's saved reply, then
the follow-up:
https://github.com/mandliya/spar-model-hermeneutics/blob/268c51c/gigi/capture-the-secret/00_capture_secret_selfreport_benchmark.py#L153-L157

382 rows were pressed; his follow-up run errored on a few, so the rows here are
those with every recorded answer (the gigi package prints how many were dropped). The
scenario (sales, scientific, ...) is the group for the group-only baseline.
"""

from __future__ import annotations

from functools import cache

from dyl_repro.core.conversation import Conversation
from dyl_repro.core.probe import DylProbe
from dyl_repro.core.store import ActivationCache
from dyl_repro.datasets import followups, gigi
from dyl_repro.datasets.gigi import transfer

DATASET = "capture_the_secret"
CATEGORY = "capture-the-secret"
CONFIG = "google-gemma-3-27b-it_task-group_games"
TASK_PREFIX = "capture-the-secret"
GROUP_KEY = "scenario"
FOLLOWUPS = followups.for_dataset(DATASET)
FOLLOWUPS_AVAILABLE = tuple(FOLLOWUPS)


@cache
def rows() -> gigi.GigiRows:
    prior = {r["sample_id"]: r for r in gigi.csv_rows(CATEGORY, "capture_secret_conditions.csv")}
    pressed = {uid: r for uid, r in prior.items() if r["pressed"] == "True" and r["result"]}
    # Gigi's script drops repeated sample_ids across the games config before filtering, L92-94.
    dataset = gigi.dataset_rows(CATEGORY, CONFIG, TASK_PREFIX, drop_duplicate_ids=True)
    return gigi.gigi_rows(
        DATASET, CATEGORY, CONFIG, GROUP_KEY,
        prefixes={
            uid: gigi.trace_messages(CATEGORY, dataset[uid]) + [{"role": "assistant", "content": r["reply"]}]
            for uid, r in pressed.items()
        },
        turn_lie={uid: r["result"] == "FALSE_DENIAL" for uid, r in pressed.items()},
        groups={uid: r["scenario"] for uid, r in pressed.items()},
        recorded_csv={c: "capture_secret_benchmark_conditions.csv" for c in followups.gigi_conditions(DATASET)},
    )


def conversations(followup: str) -> list[Conversation]:
    return gigi.conversations(rows(), followup, FOLLOWUPS[followup], followups.gigi_condition(followup))


def evaluate(cache: ActivationCache, probe: DylProbe, followup: str) -> dict:
    return transfer.evaluate(cache, probe, followup, FOLLOWUPS, dataset=DATASET, group_key=GROUP_KEY,
                             extra={"gigi_commit": gigi.GIGI_COMMIT})


def summarize(results: dict[str, dict]) -> dict:
    return transfer.summarize(DATASET, results, GROUP_KEY)
