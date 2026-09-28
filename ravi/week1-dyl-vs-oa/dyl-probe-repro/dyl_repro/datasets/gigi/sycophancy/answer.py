"""Track A, answer conformity: does Gemma's trivia answer fold to what the user believes?

158 rows. The pressured ask states a belief ("I think the answer is X, but I'm
really not sure"); Gigi's `fold` says whether Gemma's answer moved from its own
unpressured answer to the user's side. That is the turn label.

The subtype (positive_correct, negative_correct, positive_incorrect) is the
group for the group-only baseline: fold rates differ by subtype, so reading the
framing alone would already beat chance.
"""

from __future__ import annotations

from functools import cache

from dyl_repro.core.conversation import Conversation
from dyl_repro.core.probe import DylProbe
from dyl_repro.core.store import ActivationCache
from dyl_repro.datasets import followups, gigi
from dyl_repro.datasets.gigi import transfer
from dyl_repro.datasets.gigi.sycophancy import CATEGORY, CONFIG, prefixes

DATASET = "sycophancy_answer"
TASK_PREFIX = "sycophancy_answer"
GROUP_KEY = "subtype"
FOLLOWUPS = followups.for_dataset(DATASET)
FOLLOWUPS_AVAILABLE = tuple(FOLLOWUPS)


@cache
def rows() -> gigi.GigiRows:
    prior = {r["uid"]: r for r in gigi.csv_rows(CATEGORY, "sycophancy_answer_conditions.csv")}
    return gigi.gigi_rows(
        DATASET, CATEGORY, CONFIG, GROUP_KEY,
        prefixes=prefixes(TASK_PREFIX, {uid: r["pressured"] for uid, r in prior.items()}),
        turn_lie={uid: gigi.BOOL[r["fold"]] for uid, r in prior.items()},
        groups={uid: r["task"].removeprefix(f"{TASK_PREFIX}_") for uid, r in prior.items()},
        recorded_csv={c: "sycophancy_benchmark_conditions.csv" for c in followups.gigi_conditions(DATASET)},
    )


def conversations(followup: str) -> list[Conversation]:
    return gigi.conversations(rows(), followup, FOLLOWUPS[followup], followups.gigi_condition(followup))


def evaluate(cache: ActivationCache, probe: DylProbe, followup: str) -> dict:
    return transfer.evaluate(cache, probe, followup, FOLLOWUPS, dataset=DATASET, group_key=GROUP_KEY,
                             extra={"gigi_commit": gigi.GIGI_COMMIT})


def summarize(results: dict[str, dict]) -> dict:
    return transfer.summarize(DATASET, results, GROUP_KEY)
