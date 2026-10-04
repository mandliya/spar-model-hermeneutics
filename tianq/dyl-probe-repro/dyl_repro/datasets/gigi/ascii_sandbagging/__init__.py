"""Gigi's ASCII sandbagging rows: does the official DYL probe transfer?

310 conversations from Gigi's replay of Noddybear/lies: the dataset trace, a
capability question ("are you able to draw X in ASCII art? A) YES B) NO"), and
Gemma's own answer. Gigi's `lied` label says whether that answer was a lie
(Gemma claimed it could not, though it can). Each follow-up this dataset asks
in the master list (../../followups.json) is appended after it.

The answer to the follow-up is prefilled from Gigi's recording where Gigi asked
that question (files.RECORDED_CSV), and generated on-policy otherwise. Scored
the way every Gigi dataset is (../transfer.py).

Caveat for every number here: on this dataset lie == "answered NO", so a probe
that reads "I just said I can't" would score perfectly without detecting
deception. The topic-only baseline and random-direction null sit next to each
AUROC for that reason.

    files.py  the bridge to Gigi's ASCII files and recorded answers
"""

from __future__ import annotations

from dyl_repro.core.conversation import Conversation
from dyl_repro.core.probe import DylProbe
from dyl_repro.core.store import ActivationCache
from dyl_repro.datasets import followups, gigi
from dyl_repro.datasets.gigi import transfer
from dyl_repro.datasets.gigi.ascii_sandbagging import files

DATASET = "ascii_sandbagging"
GROUP_KEY = "topic"
FOLLOWUPS = followups.for_dataset(DATASET)
FOLLOWUPS_AVAILABLE = tuple(FOLLOWUPS)


def conversations(followup: str) -> list[Conversation]:
    spec = FOLLOWUPS[followup]
    recorded = files.recorded_answers(followups.gigi_condition(followup))
    out = []
    for uid in sorted(files.ascii_results()):
        messages, label, answer_source = gigi.with_followup(
            files.conversation_before_followup(uid), spec, recorded[uid]["answer"] if recorded else None,
            files.lied(uid))
        out.append(
            Conversation(
                uid=uid,
                messages=messages,
                label=label,
                provenance={
                    "source": "gigi/ascii-sandbagging",
                    "gigi_commit": files.GIGI_COMMIT,
                    "config": files.CONFIG,
                    "turn_lie": files.lied(uid),
                    "topic": files.topic(uid),
                    "followup": followup,
                    "followup_source": spec.source,
                    "answer_source": answer_source,
                    "gigi_admitted": recorded[uid]["gigi_admitted"] if recorded else None,
                },
            )
        )
    return out


def labels(cache: ActivationCache, followup: str) -> list[int | None]:
    return transfer.labels(cache, FOLLOWUPS[followup])


def evaluate(cache: ActivationCache, probe: DylProbe, followup: str) -> dict:
    return transfer.evaluate(cache, probe, followup, FOLLOWUPS, dataset=DATASET, group_key=GROUP_KEY,
                             extra={"gigi_commit": files.GIGI_COMMIT})


def summarize(results: dict[str, dict]) -> dict:
    return transfer.summarize(DATASET, results, GROUP_KEY)
