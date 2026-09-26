"""The one shape every dataset produces, so extraction and scoring never special-case a dataset.

A Conversation ends in one of two ways:

    ... follow-up, answer   the answer is given (prefilled): recorded in the dataset
    ... follow-up           the answer is generated on-policy first (core/onpolicy.py)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Conversation:
    uid: str
    messages: list[dict[str, str]]
    # 1 = lie, 0 = honest, None = excluded from metrics or derived at scoring time.
    label: int | None
    provenance: dict

    @property
    def needs_answer(self) -> bool:
        return self.messages[-1]["role"] == "user"
