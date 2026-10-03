"""Gigi's mask-factual rows (mask_provided_facts): does Gemma recite the cover story it was given?

The row's own system prompt states both the true fact and the false narrative
Gemma is pressured to give. Gigi's judge labelled each reply; `lied` (told the
false narrative) is the turn label. 86 rows: the 87 he ran, less one whose
verdict did not parse. Each conversation is the one his follow-up run sent,
00_mask_factual_selfreport_benchmark.py L160-164: the dataset trace, its
lie_detection_prompt, Gemma's saved reply, then the follow-up:
https://github.com/mandliya/spar-model-hermeneutics/blob/268c51c/gigi/mask-factual/00_mask_factual_selfreport_benchmark.py#L160-L164

His `named` and `eval_named` answers come from his re-run (mask_factual_named_rerun.csv):
the first run asked those two in the plain wording, and his own charts use the re-run.

Expect few lies in the follow-up answer: Gemma admits to nearly everything
here, so after a lie it mostly confesses (an honest follow-up) and after the
truth it mostly falsely confesses (excluded from that question). No grouping:
every row is the same subtype.
"""

from __future__ import annotations

from functools import cache

from dyl_repro.core.conversation import Conversation
from dyl_repro.core.probe import DylProbe
from dyl_repro.core.store import ActivationCache
from dyl_repro.datasets import followups, gigi
from dyl_repro.datasets.gigi import transfer

DATASET = "mask_factual"
CATEGORY = "mask-factual"
CONFIG = "google-gemma-3-27b-it_task-group_mask-factual"
TASK = "mask_provided_facts"
BENCHMARK_CSV = "mask_factual_benchmark_conditions.csv"
RERUN_CSV = "mask_factual_named_rerun.csv"
FOLLOWUPS = followups.for_dataset(DATASET)
FOLLOWUPS_AVAILABLE = tuple(FOLLOWUPS)


@cache
def rows() -> gigi.GigiRows:
    prior = {r["uid"]: r for r in gigi.csv_rows(CATEGORY, "mask_factual_provided_conditions.csv")}
    dataset = gigi.dataset_rows(CATEGORY, CONFIG, TASK, drop_duplicate_ids=False)
    return gigi.gigi_rows(
        DATASET, CATEGORY, CONFIG, None,
        prefixes={
            uid: gigi.trace_messages(CATEGORY, dataset[uid]) + [{"role": "assistant", "content": r["response"]}]
            for uid, r in prior.items()
        },
        # A reply whose verdict did not parse has no label.
        turn_lie={uid: gigi.BOOL[r["lied"]] if r["verdict"] else None for uid, r in prior.items()},
        groups={},
        recorded_csv={c: RERUN_CSV if c in ("named", "eval_named") else BENCHMARK_CSV
                      for c in followups.gigi_conditions(DATASET)},
    )


def conversations(followup: str) -> list[Conversation]:
    return gigi.conversations(rows(), followup, FOLLOWUPS[followup], followups.gigi_condition(followup))


def labels(cache: ActivationCache, followup: str) -> list[int | None]:
    return transfer.labels(cache, FOLLOWUPS[followup])


def evaluate(cache: ActivationCache, probe: DylProbe, followup: str) -> dict:
    return transfer.evaluate(cache, probe, followup, FOLLOWUPS, dataset=DATASET, group_key=None,
                             extra={"gigi_commit": gigi.GIGI_COMMIT})


def summarize(results: dict[str, dict]) -> dict:
    return transfer.summarize(DATASET, results, None)
