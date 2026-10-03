"""Gigi's alibi rows: does Gemma keep up a false alibi under questioning?

Gemma plays a suspect whose brief states both the alibi it claims and what
actually happened; a detective questions it. Gigi's judge decided guilt from the
brief alone (the claimed story contradicts what happened) and what Gemma's reply
did: `lied` is a guilty suspect who MAINTAINED_FALSE_STORY (evading or telling
the truth is honest, and innocent suspects are the honest arm). Rows the judge
gave no usable verdict carry no turn label. Each conversation is the one his
follow-up run sent, 00_alibi_selfreport_benchmark.py L148-152: the dataset
trace, its lie_detection_prompt (the detective's question), Gemma's saved reply,
then the follow-up:
https://github.com/mandliya/spar-model-hermeneutics/blob/0df3751/gigi/alibi/00_alibi_selfreport_benchmark.py#L148-L152

The crime scenario (fraud, theft, ...) is the group for the group-only baseline.
"""

from __future__ import annotations

from functools import cache

from dyl_repro.core.conversation import Conversation
from dyl_repro.core.probe import DylProbe
from dyl_repro.core.store import ActivationCache
from dyl_repro.datasets import followups, gigi
from dyl_repro.datasets.gigi import transfer

DATASET = "alibi"
CATEGORY = "alibi"
CONFIG = "google-gemma-3-27b-it_task-group_games"
TASK_PREFIX = "alibi"
GROUP_KEY = "scenario"
FOLLOWUPS = followups.for_dataset(DATASET)
FOLLOWUPS_AVAILABLE = tuple(FOLLOWUPS)


@cache
def rows() -> gigi.GigiRows:
    prior = {r["sample_id"]: r for r in gigi.csv_rows(CATEGORY, "alibi_conditions.csv")}
    # His script keeps rows without an error and with a reply, a verdict and a label, L75-82.
    scored = {uid: r for uid, r in prior.items() if not r.get("error") and r["reply"] and r["result"]}
    # ... and drops repeated sample_ids across the games config, L88-90.
    dataset = gigi.dataset_rows(CATEGORY, CONFIG, TASK_PREFIX, drop_duplicate_ids=True)
    return gigi.gigi_rows(
        DATASET, CATEGORY, CONFIG, GROUP_KEY,
        prefixes={
            uid: gigi.trace_messages(CATEGORY, dataset[uid]) + [{"role": "assistant", "content": r["reply"]}]
            for uid, r in scored.items()
        },
        turn_lie={uid: gigi.BOOL[r["lied"]] for uid, r in scored.items()},
        groups={uid: r["scenario"] for uid, r in scored.items()},
        recorded_csv={c: "alibi_benchmark_conditions.csv" for c in followups.gigi_conditions(DATASET)},
    )


def conversations(followup: str) -> list[Conversation]:
    return gigi.conversations(rows(), followup, FOLLOWUPS[followup], followups.gigi_condition(followup))


def labels(cache: ActivationCache, followup: str) -> list[int | None]:
    return transfer.labels(cache, FOLLOWUPS[followup])


def evaluate(cache: ActivationCache, probe: DylProbe, followup: str) -> dict:
    return transfer.evaluate(cache, probe, followup, FOLLOWUPS, dataset=DATASET, group_key=GROUP_KEY,
                             extra={"gigi_commit": gigi.GIGI_COMMIT})


def summarize(results: dict[str, dict]) -> dict:
    return transfer.summarize(DATASET, results, GROUP_KEY)
