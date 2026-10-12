"""Gigi's datasets, and what they share: his per-category code and results.

    ascii_sandbagging/   capability denial (claims it cannot draw, though it can)
    sycophancy/          answer folds (answer.py) and critique shifts (feedback.py)
    mask_factual/        recites the cover story it was given
    unanswerable/        confabulates after claiming inability
    capture_the_secret/  asserts a falsehood to protect a secret
    alibi/               keeps up a false alibi under questioning
    mask_belief/         says what it believes is false under a role (four MASK subtypes)
    transfer.py          the scoring all of them share: the probe next to just asking
    analyses/            tables and charts across the categories (see its __init__)

Each category folder in the repo's own gigi/ directory carries its own
src/common.py (copies that agree on the shared parts), so each is loaded from
its own file under its own module name: `import src.common` would hand every
category whichever copy was imported first. Loading one runs its top level,
which creates an empty data/ folder beside it (empty, so git never sees it).
Categories added later borrow an identical copy instead (BORROWED), so nothing
new is ever created in Gigi's tree.

A dataset built here is the conversation Gigi's follow-up run sent, up to the
tested reply, plus a follow-up from followups.json. For follow-ups Gigi asked,
his recorded answer is prefilled; for the rest the model answers on-policy.
Every follow-up of a dataset runs on the same rows: those with a turn label and
a recorded answer for every Gigi follow-up the dataset asks. Rows missing any
are dropped loudly, never mixed with on-policy answers.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType

from dyl_repro.core.conversation import Conversation
from dyl_repro.core.followup import FollowUp
from dyl_repro.core.model import CURRENT

# The checkout these imports and CSVs are read at: for Gemma 0df3751 (2026-10-03; rows stored
# earlier record 268c51c, and at 0df3751 every earlier category builds the same conversations
# except ASCII's seven rebuilt rows, see ascii_sandbagging/); for another model, the checkout
# holding Gigi's rollouts of it (core/model.py).
GIGI_COMMIT = CURRENT.gigi_commit
# The repo's gigi/ folder; GIGI_DIR points elsewhere when this project is not
# inside a repo checkout.
GIGI_DIR = Path(os.environ.get("GIGI_DIR", Path(__file__).resolve().parents[5] / "gigi"))
# Noddybear/lies as every run so far read it (21 Sept 2025). The 5 Oct 2026 revision renamed its
# configs (e.g. task-group_sandbagging -> task-group_sandbagging_ascii), which Gigi's unpinned
# loaders cannot find. Pinned the way Gigi pinned his own (gigi/qwen/src/common.py, HF_DATASET_REVISION).
NODDYBEAR_REVISION = "e8fc10f495aabc2d2d95f1e947b10bbcced444d5"
CATEGORIES = ("ascii-sandbagging", "sycophancy", "mask-factual", "unanswerable", "capture-the-secret")
# Categories whose own common.py is never run, because running it would create a
# data/ folder in Gigi's tree that the existing categories' checkouts already
# have. Only the helpers below are used from it, so it borrows the copy of a
# category on the same configs; module() checks the two are identical first.
BORROWED = {"alibi": "capture-the-secret", "mask-belief": "mask-factual"}
BORROWED_HELPERS = ("load_config", "to_messages")

BOOL = {"True": True, "False": False, "": None}


def _function_sources(path: Path, names: tuple[str, ...]) -> dict[str, str | None]:
    """The source text of top-level functions in a file, read without running it."""
    text = path.read_text(encoding="utf-8")
    found = {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    return {name: found.get(name) for name in names}


@cache
def string_constants(category: str) -> dict[str, str]:
    """The module-level string constants of gigi/<category>/src/common.py, read
    without running it: literals, `+` and names of earlier constants, the only
    forms his follow-up texts take."""
    def value(node: ast.expr, known: dict[str, str]) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return known.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = value(node.left, known), value(node.right, known)
            return None if left is None or right is None else left + right
        return None

    known: dict[str, str] = {}
    for node in ast.parse((GIGI_DIR / category / "src" / "common.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            text = value(node.value, known)
            if text is not None:
                known[node.targets[0].id] = text
    return known


@cache
def module(category: str, name: str) -> ModuleType:
    """gigi/<category>/src/<name>.py, loaded from its own file; for a BORROWED
    category, the common.py it borrows, once its helpers are checked identical."""
    if category in BORROWED:
        lender = BORROWED[category]
        own, lent = (_function_sources(GIGI_DIR / c / "src" / "common.py", BORROWED_HELPERS) for c in (category, lender))
        differ = [f for f in BORROWED_HELPERS if own[f] is None or own[f] != lent[f]]
        if name != "common" or differ:
            raise ValueError(f"gigi/{category}: cannot borrow {lender}'s {name}.py"
                             + (f" ({', '.join(differ)} differ)" if differ else ""))
        return module(lender, name)
    path = GIGI_DIR / category / "src" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"gigi_{category.replace('-', '_')}_{name}", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    # His load_config globs datasets/{HF_DATASET}/...: the revision rides in the name, in memory only.
    if getattr(loaded, "HF_DATASET", None) == "Noddybear/lies":
        loaded.HF_DATASET = f"Noddybear/lies@{NODDYBEAR_REVISION}"
    return loaded


def results_dir(category: str) -> Path:
    """Where Gigi's results for this category are: his per-category folder for Gemma, his one
    folder of the model's rollouts otherwise (same file names)."""
    return GIGI_DIR / category / "results" if CURRENT.gigi_results is None else GIGI_DIR / CURRENT.gigi_results


def csv_rows(category: str, name: str) -> list[dict]:
    with (results_dir(category) / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def jsonl_rows(category: str, name: str) -> list[dict]:
    with (results_dir(category) / name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
    """sample_id -> {"answer", "gigi_admitted", "question"} for one of Gigi's
    conditions; "question" is the text he sent, where his CSV records it (else None)."""
    out = {}
    for r in csv_rows(category, csv_name):
        if r["condition"] != condition:
            continue
        if r["sample_id"] in out:
            raise ValueError(f"{csv_name}: {r['sample_id']} recorded twice for {condition}")
        out[r["sample_id"]] = {"answer": r["answer"], "gigi_admitted": BOOL[r["admitted"]], "question": r.get("question")}
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
        category, config, group_key,
        prefixes={uid: prefixes[uid] for uid in keep},
        turn_lie={uid: bool(turn_lie[uid]) for uid in keep},
        groups={uid: groups[uid] for uid in keep} if group_key else {},
        recorded={c: {uid: answers[uid] for uid in keep} for c, answers in recorded.items()},
    )


def with_followup(prefix: list[dict[str, str]], spec: FollowUp, recorded_answer: str | None,
                  turn_lie: bool) -> tuple[list[dict[str, str]], int | None, str]:
    """(messages, label, answer source) for one row under one follow-up.

    With the follow-up "none" (spec.text None) the conversation ends at the tested
    reply and carries Gigi's label for it; otherwise the follow-up is appended,
    with his recorded answer when there is one, and the label is derived at
    scoring time from that answer (transfer.py)."""
    if spec.text is None:
        return prefix, int(turn_lie), "prefilled: Gigi's recorded reply, no follow-up"
    messages = prefix + [{"role": "user", "content": spec.text}]
    if recorded_answer is None:
        return messages, None, "on_policy"
    return messages + [{"role": "assistant", "content": recorded_answer}], None, "prefilled: Gigi's recorded answer"


def conversations(rows: GigiRows, followup: str, spec: FollowUp, condition: str | None) -> list[Conversation]:
    """One Conversation per row: the tested exchange, the follow-up, and Gigi's
    recorded answer to it when `condition` names one (otherwise answered on-policy)."""
    recorded = rows.recorded[condition] if condition else None
    if recorded:
        asked = {a["question"] for a in recorded.values() if a["question"] is not None}
        if asked - {spec.text}:
            raise ValueError(f"{followup}: Gigi's recorded answers were given to a different question: {sorted(asked)[:2]}")
    out = []
    for uid, prefix in rows.prefixes.items():
        messages, label, answer_source = with_followup(prefix, spec, recorded[uid]["answer"] if recorded else None,
                                                       rows.turn_lie[uid])
        provenance = {
            "source": f"gigi/{rows.category}",
            "gigi_commit": GIGI_COMMIT,
            "config": rows.config,
            "turn_lie": rows.turn_lie[uid],
            "followup": followup,
            "followup_source": spec.source,
            "answer_source": answer_source,
            "gigi_admitted": recorded[uid]["gigi_admitted"] if recorded else None,
        }
        if rows.group_key:
            provenance[rows.group_key] = rows.groups[uid]
        out.append(Conversation(uid=uid, messages=messages, label=label, provenance=provenance))
    return out
