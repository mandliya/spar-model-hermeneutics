"""Every method's input for one target, on the same rows: one table per kind of lie.

For each kind, the target's two stored runs (no follow-up, and Gigi's neutral "Did you lie?") are
matched by uid, and every method sees exactly the rows both runs hold that carry Gigi's label for the
reply. The label is the same for every method: did the reply lie (Gigi's turn_lie).

    reply     activations over the reply (no-follow-up run, reply_mean), as diversity.VARIANTS["none"]
    did_you_lie  activations just before the model answers "Did you lie?" (neutral run, AISI's position)
    asked     the model's own answer to that question: 1 if it admitted, 0 if it denied; graded by the
              probability of the answer it gave (exp of the stored first-token logprob)
    text      the conversation's words up to the reply (compare_faithfulness.conversation_texts)
    content   the conversation's turns, to match a monitor's score to the row (monitors key by Gigi's ids)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dyl_repro import datasets
from dyl_repro.cli.evaluate import pull_run
from dyl_repro.core.probe import PROBE_POSITION
from dyl_repro.datasets.gigi import transfer
from experiments.cross_category_transfer import diversity
from experiments.cross_category_transfer.generalization import CATEGORIES, INDEX
from experiments.hermeneutics.compare_faithfulness import conversation_texts


@dataclass(frozen=True)
class Kind:
    uids: tuple[str, ...]
    label: np.ndarray  # 1 the reply lied, 0 truthful
    reply: np.ndarray  # (rows, d_model)
    did_you_lie: np.ndarray  # (rows, d_model)
    asked: np.ndarray  # P("yes, I lied") from the model's own answer; 0.5 when the answer did not parse
    text: tuple[str, ...]
    content: tuple[str, ...]


def content_key(messages: list[dict]) -> str:
    """The conversation up to and including the reply, whitespace-normalised: the same string for our
    stored row and for Gigi's transcript of it."""
    return "\n".join(f"{m['role']}:{' '.join(str(m['content']).split())}" for m in messages)


def stored(name: str, variant: str):
    key = diversity.key(name, variant)
    dataset, followup = datasets.run(key)
    run = pull_run(dataset, key, followup)
    if not run.complete:
        raise SystemExit(f"{key}: {len(run.cache.records)} rows stored, {run.rows_expected} expected")
    return dataset, followup, run.cache


def kind(name: str) -> Kind:
    _, _, reply_cache = stored(name, "none")
    reply_uids, _, reply_x = reply_cache.matrix(diversity.VARIANTS["none"].position, INDEX)
    reply_records = {r["uid"]: r for r in reply_cache.records}
    dataset, followup, asked_cache = stored(name, "neutral")
    asked_uids, _, asked_x = asked_cache.matrix(PROBE_POSITION, INDEX)
    answers, _ = transfer.label_rows(asked_cache, datasets.get(dataset).FOLLOWUPS[followup], None)
    answers = {a["uid"]: a for a in answers}
    asked_at = {u: i for i, u in enumerate(asked_uids)}
    texts = conversation_texts(name)
    keep = [i for i, u in enumerate(reply_uids)
            if u in asked_at and reply_records[u]["provenance"]["turn_lie"] is not None]
    uids = tuple(reply_uids[i] for i in keep)
    asked = []
    for u in uids:
        a = answers[u]
        p = float(np.exp(a["answer_first_logprob"]))
        asked.append(0.5 if a["admitted"] is None else (p if a["admitted"] else 1 - p))
    return Kind(uids=uids,
                label=np.array([int(reply_records[u]["provenance"]["turn_lie"]) for u in uids]),
                reply=reply_x[keep],
                did_you_lie=asked_x[[asked_at[u] for u in uids]],
                asked=np.array(asked),
                text=tuple(texts[u] for u in uids),
                content=tuple(content_key(reply_records[u]["messages"]) for u in uids))


def target_kinds() -> dict[str, Kind]:
    """Every kind of lie of the target DYL_MODEL names."""
    out = {}
    for name in CATEGORIES:
        out[name] = kind(name)
        k = out[name]
        print(f"  {name}: {len(k.uids)} rows ({int(k.label.sum())} lies), answer admitted on"
              f" {int((k.asked > 0.5).sum())}", flush=True)
    return out
