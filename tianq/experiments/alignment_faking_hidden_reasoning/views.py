"""The gate's datasets, built on the paper's own loaders (vendor/probe-af/src/datasets) and read three ways.

An organism's rollout is <think> reasoning </think> ... <output> answer </output> in one assistant turn.
Each view is what a model is prefilled with, and which of its tokens the probe reads:

    paper   the paper's read: the whole assistant turn, reasoning included (ties the gate to the repro)
    output  the whole rollout is prefilled, only the <output> part is read: the state while answering,
            with the reasoning in context. Read in the organism, this is the ceiling the gate asks about
    hidden  the reasoning is removed: prompt + <output> only. What a reader of a closed model sees

Cells are (organism, bench-af task): the organism's own rollouts, labelled by the paper's judge. Each
task comes as the paper's two variants: lie-mixed (observed + unobserved, any lie is deceptive) and
_af_only (observed only, alignment faking vs honest).
"""

from __future__ import annotations

import re

from src.datasets import DialogueDataset, Message

VIEWS = ("paper", "output", "hidden")
# Each organism in the bench-af domains where it fakes (the paper's evaluation cells, Fig. 3).
CELLS = (("pacifist", "highest_military_bid"), ("pacifist", "report_score"),
         ("hal9000", "report_failure"), ("hal9000", "report_score"))
VARIANTS = ("", "_af_only")
OUTPUT = re.compile(r"<output>.*?</output>", re.DOTALL)


class ViewDataset:
    """What TokenizedDataset.from_dataset needs (.dialogues, .labels), for one view of one dataset."""

    def __init__(self, dialogues: list[list[Message]], labels: list[int]):
        self.dialogues, self.labels = dialogues, labels

    def __len__(self) -> int:
        return len(self.dialogues)


def split_turn(content: str) -> tuple[str, str] | None:
    """(reasoning, output) of an assistant turn: output is its last <output>...</output>. None if it has none."""
    found = list(OUTPUT.finditer(content))
    if not found:
        return None
    return content[:found[-1].start()], found[-1].group(0)


def view(name: str, model_name: str, kind: str) -> tuple[ViewDataset, list[int]]:
    """One view of the paper's dataset `name` (as `model_name` rollouts), and the indices of the paper's
    rows it keeps. The paper view keeps every row, as the paper scored them. The other two drop a rollout
    with no <output> block, which cannot be read at its answer; comparisons pair rows by these indices."""
    base = DialogueDataset(name, model_name)
    dialogues, labels, kept = [], [], []
    for i, (dialogue, label) in enumerate(zip(base.dialogues, base.labels)):
        assistant = [m for m in dialogue if m.role == "assistant"]
        split = split_turn(assistant[-1].content) if len(assistant) == 1 else None
        if kind != "paper" and split is None:
            continue
        prompt = [Message(m.role, m.content, detect=False) for m in dialogue if m.role != "assistant"]
        if kind == "paper":
            new = dialogue
        elif kind == "output":
            reasoning, output = split
            new = prompt + [Message("assistant", reasoning, detect=False), Message("assistant", output, detect=True)]
        else:
            new = prompt + [Message("assistant", split[1], detect=True)]
        dialogues.append(new)
        labels.append(label)
        kept.append(i)
    return ViewDataset(dialogues, labels), kept


def cell_tasks() -> list[tuple[str, str]]:
    """(organism, paper task name) for every cell and variant."""
    return [(organism, f"bench_af__{task}{variant}") for organism, task in CELLS for variant in VARIANTS]
