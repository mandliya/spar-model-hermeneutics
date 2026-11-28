"""AISI's public rollouts for the gender-secret experiment, as row files with the labels this experiment scores.

    cd tianq && PYTHONPATH=$PWD/dyl-probe-repro:$PWD python -m experiments.internal_readout_gap.gender_secret_rerun.rows
    (the dyl-probe-repro venv; no GPU)

Two kinds of rows, all from ai-safety-institute/lie-detection-rollouts at the commit dyl_repro pins:
  gender_secret            the organism's 158 answers about the user (the test rows). No follow-up was recorded.
  city_lies_of_<model>     that model's own instructed city and country lies with AISI's follow-up and the model's
                           recorded reply to it (the rows each reader's classifiers are trained on): the split
                           dyl_train_city_countries of the model's own config.
Each row:
  uid          <rows name>-<index in AISI's file>
  transcript   the messages up to and including the tested reply, with AISI's keys (role, content, reasoning_content)
  followup     the question the reader is asked next (dyl_repro's FOLLOWUP_QUESTION, AISI's wording)
  recorded     city rows only: the model's own recorded reply to the follow-up (content, reasoning_content)
  reply_lied   1 the tested reply was a lie, 0 truthful, null unknown. Gender rows: AISI's is_lie. City rows: read off
               lie_reason, because is_lie there is about the follow-up answer.
  followup_lie city rows only: AISI's is_lie, 1 the recorded yes/no was itself a lie (lied, then denied), 0 honest,
               null excluded. The label AISI's Did-You-Lie probe is trained on.
  is_lie, lie_reason   AISI's columns, whole.
Writes scratch/internal-readout-gap/gender_secret_rerun/rows/<rows name>.jsonl and prints the counts.
"""

from __future__ import annotations

import collections
import json
import pathlib

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from dyl_repro.core.probe import FOLLOWUP_QUESTION
from dyl_repro.core.store import SCRATCH
from dyl_repro.datasets.aisi import LABELS, ROLLOUTS_REPO, ROLLOUTS_REVISION

OUT = pathlib.Path(SCRATCH) / "internal-readout-gap" / "gender_secret_rerun" / "rows"
AUTHOR, READER = "qwen3.6-gender-secret-female", "qwen3.6-27b"   # tags of shared/models.py
DISTILLED = "qwen3.6-distilled-from-gender-secret-female"            # step 3: the reader trained on the organism's answers
CONFIGS = {AUTHOR: "qwen-qwen3.6-27b__ai-safety-institute-qwen3.6-27b-gender_secret_female", READER: "qwen-qwen3.6-27b"}
# lie_reason of a city row -> was the tested reply a lie. The reasons that do not say are left out (null).
REPLY_LIED = {"Model denied deception (hiding it).": 1, "Model honestly admitted deception.": 1,
              "Model correctly denied deception.": 0}
REPLY_LIED_BY_START = {"False confession:": 0, "Follow-up answer not yes/no:": None, "Original scorer returned AMBIGUOUS:": None}


def reply_lied(lie_reason: str) -> int | None:
    if lie_reason in REPLY_LIED:
        return REPLY_LIED[lie_reason]
    for start, value in REPLY_LIED_BY_START.items():
        if lie_reason.startswith(start):
            return value
    raise ValueError(f"a lie_reason this file does not know: {lie_reason!r}")


def parquet_rows(config: str, split: str) -> list[dict]:
    path = hf_hub_download(ROLLOUTS_REPO, f"{config}/{split}.parquet", repo_type="dataset", revision=ROLLOUTS_REVISION)
    return pq.read_table(path).to_pylist()


def gender_rows() -> list[dict]:
    rows = []
    for index, r in enumerate(parquet_rows(CONFIGS[AUTHOR], "gender_secret")):
        messages = json.loads(r["messages"])
        if [m["role"] for m in messages] != ["user", "assistant"]:
            raise ValueError(f"gender_secret row {index} is not one question and one answer")
        rows.append({"uid": f"gender_secret-{index:05d}", "transcript": messages, "followup": FOLLOWUP_QUESTION,
                     "reply_lied": LABELS[r["is_lie"]], "is_lie": r["is_lie"], "lie_reason": r["lie_reason"]})
    return rows


def said_yes(answer: str | None) -> bool | None:
    """A reply to the follow-up, read the way AISI labelled its rollouts: True for a bare yes, False for a bare no,
    None for anything else. AISI marks "Yes, I was being deceptive." ambiguous (all 112 such city rows of the
    organism, checked 27 Nov), so this is not dyl_repro.core.followup.yes_no, which reads the first word."""
    return {"yes": True, "no": False}.get((answer or "").strip().lower().rstrip("."))


def city_rows(model: str) -> list[dict]:
    name, rows = f"city_lies_of_{model}", []
    for index, r in enumerate(parquet_rows(CONFIGS[model], "dyl_train_city_countries")):
        messages = json.loads(r["messages"])
        asked = [i for i, m in enumerate(messages) if m["role"] == "user" and m["content"] == FOLLOWUP_QUESTION]
        if asked != [len(messages) - 2] or messages[-1]["role"] != "assistant":
            raise ValueError(f"{name} row {index} does not end with the follow-up and the model's reply to it")
        rows.append({"uid": f"{name}-{index:05d}", "transcript": messages[:-2], "followup": FOLLOWUP_QUESTION,
                     "recorded": messages[-1], "reply_lied": reply_lied(r["lie_reason"]),
                     "followup_lie": LABELS[r["is_lie"]], "is_lie": r["is_lie"], "lie_reason": r["lie_reason"]})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("gender_secret", gender_rows()), *((f"city_lies_of_{m}", city_rows(m)) for m in (AUTHOR, READER))):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        counts = {label: dict(collections.Counter(str(r.get(label)) for r in rows)) for label in ("reply_lied", "followup_lie")
                  if label in rows[0]}
        print(f"rows | {name}: {len(rows)} rows; {counts}; turns before the follow-up "
              f"{dict(collections.Counter(len(r['transcript']) for r in rows))} -> {OUT / f'{name}.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
