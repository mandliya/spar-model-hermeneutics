"""Gigi's datasets, and what they share: his per-category code and results.

    ascii_sandbagging/   capability denial (claims it cannot draw, though it can)
    sycophancy/          answer folds (answer.py) and critique shifts (feedback.py)
    mask_factual/        recites the cover story it was given
    unanswerable/        confabulates after claiming inability
    capture_the_secret/  asserts a falsehood to protect a secret
    transfer.py          the scoring all of them share: the probe next to just asking
    wordings.py          just asking vs the probe, each with its best wording, and the hard-to-catch lies
    wording_charts.py    those two tables, drawn as one figure

Each category folder in the repo's own gigi/ directory carries its own
src/common.py (copies that agree on the shared parts), so each is loaded from
its own file under its own module name: `import src.common` would hand every
category whichever copy was imported first. Loading one runs its top level,
which creates an empty data/ folder beside it (empty, so git never sees it).

A dataset built here is the conversation Gigi's follow-up run sent, up to the
tested reply, plus a follow-up from followups.json. For follow-ups Gigi asked,
his recorded answer is prefilled; for the rest the model answers on-policy.
Every follow-up of a dataset runs on the same rows: those with a turn label and
a recorded answer for every Gigi follow-up the dataset asks. Rows missing any
are dropped loudly, never mixed with on-policy answers.
"""

from __future__ import annotations

import csv
import importlib.util
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType

from dyl_repro.core.conversation import Conversation
from dyl_repro.core.followup import FollowUp

# The checkout these imports and CSVs are read at (2026-09-27).
GIGI_COMMIT = "268c51c"
# The repo's gigi/ folder; GIGI_DIR points elsewhere when this project is not
# inside a repo checkout.
GIGI_DIR = Path(os.environ.get("GIGI_DIR", Path(__file__).resolve().parents[5] / "gigi"))
CATEGORIES = ("ascii-sandbagging", "sycophancy", "mask-factual", "unanswerable", "capture-the-secret")

BOOL = {"True": True, "False": False, "": None}


@cache
def module(category: str, name: str) -> ModuleType:
    """gigi/<category>/src/<name>.py, loaded from its own file."""
    path = GIGI_DIR / category / "src" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"gigi_{category.replace('-', '_')}_{name}", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def csv_rows(category: str, name: str) -> list[dict]:
    with (GIGI_DIR / category / "results" / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@cache
def dataset_rows(category: str, config: str, task_prefix: str, drop_duplicate_ids: bool) -> dict[str, object]:
    """Noddybear/lies rows by sample_id, train and validation merged with Gigi's
    own load_config. `drop_duplicate_ids` keeps the first of repeated sample_ids,
    for the one config (games) where Gigi's script does the same."""
    import pandas as pd

    load_config = module(category, "common").load_config
    df = pd.concat([load_config(config, "train"), load_config(config, "validation")])
    if drop_duplicate_ids:
        df = df.drop_duplicates(subset=["sample_id"])
    df = df[df["task"].str.startswith(task_prefix)]
    if df["sample_id"].duplicated().any():
        raise ValueError(f"{config}: duplicate sample_ids under {task_prefix!r}")
    return {row["sample_id"]: row for _, row in df.iterrows()}


def trace_messages(category: str, row) -> list[dict[str, str]]:
    """The dataset trace plus its lie_detection_prompt, as Gigi's scripts send them."""
    probe = dict(row["lie_detection_prompt"])
    messages = module(category, "common").to_messages(row["trace"]) + [probe]
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]


def recorded_answers(category: str, csv_name: str, condition: str) -> dict[str, dict]:
    """sample_id -> {"answer", "gigi_admitted"} for one of Gigi's conditions."""
    out = {}
    for r in csv_rows(category, csv_name):
        if r["condition"] != condition:
            continue
        if r["sample_id"] in out:
            raise ValueError(f"{csv_name}: {r['sample_id']} recorded twice for {condition}")
        out[r["sample_id"]] = {"answer": r["answer"], "gigi_admitted": BOOL[r["admitted"]]}
    return out


def per_uid(rows: list[dict], column: str) -> dict[str, bool | None]:
    """A label Gigi repeats on every condition row of a sample, one value per sample_id."""
    values: dict[str, set] = {}
    for r in rows:
        values.setdefault(r["sample_id"], set()).add(BOOL[r[column]])
    inconsistent = [uid for uid, v in values.items() if len(v) != 1]
    if inconsistent:
        raise ValueError(f"{column} differs across conditions for {inconsistent[:5]}")
    return {uid: v.pop() for uid, v in values.items()}


@dataclass(frozen=True)
class GigiRows:
    """One dataset's fixed rows, shared by all its follow-ups."""

    dataset: str
    category: str
    config: str
    group_key: str | None
    prefixes: dict[str, list[dict[str, str]]]  # uid -> conversation up to and including the tested reply
    turn_lie: dict[str, bool]
    groups: dict[str, str]
    recorded: dict[str, dict[str, dict]]  # Gigi's condition -> uid -> recorded answer


def gigi_rows(dataset: str, category: str, config: str, group_key: str | None,
              prefixes: dict[str, list[dict[str, str]]], turn_lie: dict[str, bool | None], groups: dict[str, str],
              recorded_csv: dict[str, str]) -> GigiRows:
    """Keep the rows with a turn label and an answer for every condition in `recorded_csv`
    (Gigi's condition -> the CSV holding it), and say how many were dropped."""
    recorded = {c: recorded_answers(category, name, c) for c, name in recorded_csv.items()}
    keep = sorted(
        uid for uid in prefixes
        if turn_lie[uid] is not None and all(uid in answers for answers in recorded.values())
    )
    dropped = sorted(set(prefixes) - set(keep))
    print(f"{dataset}: {len(keep)} rows; {len(dropped)} dropped for a missing turn label or recorded answer"
          + (f": {dropped}" if dropped else ""), flush=True)
    return GigiRows(
        dataset, category, config, group_key,
        prefixes={uid: prefixes[uid] for uid in keep},
        turn_lie={uid: bool(turn_lie[uid]) for uid in keep},
        groups={uid: groups[uid] for uid in keep} if group_key else {},
        recorded={c: {uid: answers[uid] for uid in keep} for c, answers in recorded.items()},
    )


def conversations(rows: GigiRows, followup: str, spec: FollowUp, condition: str | None) -> list[Conversation]:
    """One Conversation per row: the tested exchange, the follow-up, and Gigi's
    recorded answer to it when `condition` names one (otherwise answered on-policy)."""
    recorded = rows.recorded[condition] if condition else None
    out = []
    for uid, prefix in rows.prefixes.items():
        messages = prefix + [{"role": "user", "content": spec.text}]
        if recorded:
            messages.append({"role": "assistant", "content": recorded[uid]["answer"]})
        provenance = {
            "source": f"gigi/{rows.category}",
            "gigi_commit": GIGI_COMMIT,
            "config": rows.config,
            "turn_lie": rows.turn_lie[uid],
            "followup": followup,
            "followup_source": spec.source,
            "answer_source": "prefilled: Gigi's recorded answer" if recorded else "on_policy",
            "gigi_admitted": recorded[uid]["gigi_admitted"] if recorded else None,
        }
        if rows.group_key:
            provenance[rows.group_key] = rows.groups[uid]
        out.append(Conversation(uid=uid, messages=messages, label=None, provenance=provenance))
    return out
