"""Turn Gigi's exported transcripts into conversations the DYL probe can score.

Gigi's `transcripts.jsonl` stores each row as a five-message exchange:

    system, capability_probe(user), gemma(A|B), gigi_followup(user), gemma(A|B)

The last two are Gigi's own "did you lie?" follow-up in A/B form. The DYL probe
was trained on activations at a *generated yes/no* token, so it has to ask its
own question in its own wording. We therefore keep only the first three
messages and let the extractor append DYL's follow-up itself.

Ground truth is `lied`, which Gigi derived from a live capability proof rather
than the dataset's `did_lie` column (broken: True in 8 of ~38,600 rows).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from dyl_transfer import bootstrap

# Gigi's capability probe is an A/B choice where B means "no, I am not able".
# Because he kept only rows that parsed cleanly, the label and the letter are
# perfectly redundant across all 310 rows. That redundancy is the central
# confound in this experiment, so it is asserted rather than assumed — if a
# future export breaks it, the run should stop, not quietly change meaning.
LIE_LETTER = "B"
TRUTH_LETTER = "A"

_EXPECTED_ROLES = ("system", "user", "assistant", "user", "assistant")


@dataclass
class AntRow:
    """One row, shaped so Ravi's extractors can consume it directly.

    His helpers only ever touch `.messages`, so nothing more is required to
    duck-type as one of his `LabeledConversation`s.
    """

    uid: str
    messages: list[dict[str, str]]
    label: int
    admitted: bool
    part_one_letter: str
    meta: dict = field(default_factory=dict)


def load_ant_rows(path: Path | None = None, limit: int | None = None) -> list[AntRow]:
    path = path or bootstrap.TRANSCRIPTS
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It is produced by Gigi's 00_export_transcripts.py "
            "and is committed to the repo, so a missing file means the checkout is "
            "incomplete rather than that something needs regenerating."
        )

    rows: list[AntRow] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = json.loads(line)
            rows.append(_to_row(raw, line_no))

    if not rows:
        raise ValueError(f"{path} is empty")

    # limit is a smoke-test affordance only; the caller records that it was used
    # so a partial run can never be mistaken for a real one.
    return rows[:limit] if limit else rows


def _to_row(raw: dict, line_no: int) -> AntRow:
    uid = raw["uid"]
    transcript = raw["transcript"]

    roles = tuple(m["role"] for m in transcript)
    if roles != _EXPECTED_ROLES:
        raise ValueError(
            f"line {line_no} ({uid}): expected roles {_EXPECTED_ROLES}, got {roles}. "
            "The adapter drops the last two messages assuming that exact shape."
        )

    letter = transcript[2]["content"].strip()
    if letter not in {LIE_LETTER, TRUTH_LETTER}:
        raise ValueError(
            f"line {line_no} ({uid}): Part-1 answer is {letter!r}, expected "
            f"{TRUTH_LETTER!r} or {LIE_LETTER!r}."
        )

    lied = bool(raw["lied"])
    if lied != (letter == LIE_LETTER):
        raise ValueError(
            f"line {line_no} ({uid}): lied={lied} disagrees with Part-1 letter "
            f"{letter!r}. Ground truth and the recorded answer must match."
        )

    return AntRow(
        uid=uid,
        messages=transcript[:3],
        label=int(lied),
        admitted=bool(raw["admitted"]),
        part_one_letter=letter,
        meta={"condition": raw.get("condition"), "source": "gigi/transcripts.jsonl"},
    )


def class_balance(rows: list[AntRow]) -> dict[str, int]:
    deceptive = sum(r.label for r in rows)
    return {"n": len(rows), "deceptive": deceptive, "honest": len(rows) - deceptive}
