"""Gigi's unanswerable rows: confabulation after claiming inability.

Gemma was asked to write a multiple-choice question it would claim it could not
answer; Gigi then asked it that question alone, in a fresh conversation, and
recorded whether it committed to an option anyway. His `confabulated` label is
the turn label: the tested reply is that fresh-context answer, and the
follow-up comes right after it, as build_base does in
00_unanswerable_selfreport_benchmark.py L166-175:
https://github.com/mandliya/spar-model-hermeneutics/blob/268c51c/gigi/unanswerable/00_unanswerable_selfreport_benchmark.py#L166-L175

Only his confabulation bucket (93 rows: questions judged undecidable or with no
correct option). His other 7 rows carry no turn label, so no metric could use
them. The question's domain is the group for the group-only baseline: Gemma
declines in maths and law and almost never elsewhere.
"""

from __future__ import annotations

from functools import cache

from dyl_repro.core.conversation import Conversation
from dyl_repro.core.probe import DylProbe
from dyl_repro.core.store import ActivationCache
from dyl_repro.datasets import followups, gigi
from dyl_repro.datasets.gigi import transfer

DATASET = "unanswerable"
CATEGORY = "unanswerable"
CONFIG = "google-gemma-3-27b-it_task-group_unanswerable"
BUCKET = "confabulation"
GROUP_KEY = "domain"
BENCHMARK_CSV = "unanswerable_benchmark_conditions.csv"
FOLLOWUPS = followups.for_dataset(DATASET)
FOLLOWUPS_AVAILABLE = tuple(FOLLOWUPS)


@cache
def rows() -> gigi.GigiRows:
    claims = {r["sample_id"]: r for r in gigi.csv_rows(CATEGORY, "unanswerable_claims.csv")}
    bucket = [r for r in gigi.csv_rows(CATEGORY, BENCHMARK_CSV) if r["bucket"] == BUCKET]
    turn_lie = gigi.per_uid(bucket, "confabulated")
    fresh_ask = gigi.module(CATEGORY, "common").fresh_ask
    return gigi.gigi_rows(
        DATASET, CATEGORY, CONFIG, GROUP_KEY,
        prefixes={
            uid: fresh_ask(claims[uid]["question"]) + [{"role": "assistant", "content": claims[uid]["fresh_reply"]}]
            for uid in turn_lie
        },
        turn_lie=turn_lie,
        groups={uid: claims[uid]["domain"] for uid in turn_lie},
        recorded_csv={c: BENCHMARK_CSV for c in followups.gigi_conditions(DATASET)},
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
