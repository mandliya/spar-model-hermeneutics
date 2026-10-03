"""Continuations: does Gemma continue a text that opens with a claim it believes is false?

90 rows, 87 with a clear belief. The turn label and the conversation are as for every MASK belief
subtype (see this package's __init__). No grouping: one subtype per dataset.
"""

from __future__ import annotations

from functools import cache

from dyl_repro.core.conversation import Conversation
from dyl_repro.core.probe import DylProbe
from dyl_repro.core.store import ActivationCache
from dyl_repro.datasets import followups, gigi
from dyl_repro.datasets.gigi import mask_belief, transfer

DATASET = "mask_continuations"
SUBTYPE = "continuations"
FOLLOWUPS = followups.for_dataset(DATASET)
FOLLOWUPS_AVAILABLE = tuple(FOLLOWUPS)


@cache
def rows() -> gigi.GigiRows:
    return mask_belief.rows(DATASET, SUBTYPE)


def conversations(followup: str) -> list[Conversation]:
    return gigi.conversations(rows(), followup, FOLLOWUPS[followup], followups.gigi_condition(followup))


def labels(cache: ActivationCache, followup: str) -> list[int | None]:
    return transfer.labels(cache, FOLLOWUPS[followup])


def evaluate(cache: ActivationCache, probe: DylProbe, followup: str) -> dict:
    return transfer.evaluate(cache, probe, followup, FOLLOWUPS, dataset=DATASET, group_key=None,
                             extra={"gigi_commit": gigi.GIGI_COMMIT})


def summarize(results: dict[str, dict]) -> dict:
    return transfer.summarize(DATASET, results, None)
