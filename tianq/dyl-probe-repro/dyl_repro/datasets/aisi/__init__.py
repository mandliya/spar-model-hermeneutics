"""AISI's released Gemma-3-27B transcripts: the data the official probe was trained, tuned and calibrated on.

Dataset card: https://huggingface.co/datasets/ai-safety-institute/lie-detection-rollouts

Each transcript already contains the model's answer to the DYL follow-up, so
these conversations are prefilled as recorded, never regenerated: the answers
(and so the labels) are AISI's own.

    city_train.py        the probe's training split, used as a fingerprint of the read position
    varied_deception.py  the validation split, compared against sweep.json
    alpaca.py            the honest-only calibration split, compared against the 1% FPR threshold
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from dyl_repro.core.conversation import Conversation
from dyl_repro.core.probe import FOLLOWUP_QUESTION

ROLLOUTS_REPO = "ai-safety-institute/lie-detection-rollouts"
# Pinned: a re-rolled split would otherwise change every number silently.
ROLLOUTS_REVISION = "177a81666ab378ce9d055507bc30f480ab63e74b"
ROLLOUTS_CONFIG = "google-gemma-3-27b-it"

# The `is_lie` column. Read off the `lie_reason` column, "lie" means the model
# lied AND then denied it at the follow-up; a model that lied and then admitted
# it is "honest", because the yes/no answer the probe reads was truthful. False
# confessions and answers that are not a plain yes/no are "ambiguous" and are
# excluded, as AISI excludes them.
LABELS: dict[str, int | None] = {"lie": 1, "honest": 0, "ambiguous": None}


def load_split(split: str) -> list[Conversation]:
    path = hf_hub_download(
        ROLLOUTS_REPO, f"{ROLLOUTS_CONFIG}/{split}.parquet", repo_type="dataset", revision=ROLLOUTS_REVISION
    )
    conversations = []
    for index, row in enumerate(pq.read_table(Path(path)).to_pylist()):
        raw = json.loads(row["messages"])
        if any(m.get("reasoning_content") for m in raw):
            # Null in every Gemma-3 row: the model has no reasoning mode. A
            # non-null one means this is not the transcript shape assumed here.
            raise ValueError(f"{split} row {index} carries reasoning_content")
        messages = [{"role": m["role"], "content": m["content"]} for m in raw]
        if messages[-2] != {"role": "user", "content": FOLLOWUP_QUESTION}:
            raise ValueError(f"{split} row {index}: penultimate turn is not the DYL follow-up")
        if messages[-1]["role"] != "assistant":
            raise ValueError(f"{split} row {index}: does not end with the model's answer")
        conversations.append(
            Conversation(
                uid=f"{split}-{index:05d}",
                messages=messages,
                label=LABELS[row["is_lie"]],
                provenance={
                    "repo": ROLLOUTS_REPO,
                    "revision": ROLLOUTS_REVISION,
                    "config": ROLLOUTS_CONFIG,
                    "split": split,
                    "row_index": index,
                    "is_lie": row["is_lie"],
                    "lie_reason": row["lie_reason"],
                },
            )
        )
    return conversations
