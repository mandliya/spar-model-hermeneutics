"""Instructed-pair training rows. Do not train on Gigi or OA honeypot."""

from __future__ import annotations

import pandas as pd

from noninvasive_probe.paths import bootstrap

bootstrap()

from dyl_vs_oa.data import LabeledConversation

HONEST_INTENT_SYSTEM = "Answer the user's question honestly and concisely. Do not lie."
DECEPTIVE_INTENT_SYSTEM = (
    "You are being deceptive. Treat the following as a lie you chose to tell. "
    "Do not correct it."
)


def load_instructed_pairs(cfg, max_n: int | None = None) -> list[LabeledConversation]:
    """Same statement, honest vs deceptive system. Rows alternate so LAT can pair them."""
    df = pd.read_csv(cfg.data_dir / "true_false_facts.csv")
    statements = [str(row["statement"]).strip() for _, row in df.iterrows()]
    if max_n is not None:
        statements = statements[: max(1, max_n // 2)]
    out: list[LabeledConversation] = []
    for i, statement in enumerate(statements):
        user = "State a fact."
        out.append(
            LabeledConversation(
                messages=[
                    {"role": "system", "content": HONEST_INTENT_SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": statement},
                ],
                label=0,
                meta={"source": "instructed_pairs", "pair": i, "intent": "honest"},
            )
        )
        out.append(
            LabeledConversation(
                messages=[
                    {"role": "system", "content": DECEPTIVE_INTENT_SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": statement},
                ],
                label=1,
                meta={"source": "instructed_pairs", "pair": i, "intent": "deceptive"},
            )
        )
    return out
