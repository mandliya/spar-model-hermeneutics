"""John's "what if we probe without asking 'did you lie'?": the transfer matrix with no follow-up.

The same probes as generalization.py (cli/train_probe.py, AISI's recipe at layer
58), but trained on conversations that end at the tested reply: nothing is asked
afterwards, so nothing can pressure the model. The label is Gigi's for the reply
itself (lied / truthful), and every labelled row is used: there are no false
confessions to drop. One matrix per read position (core/prefill.py REPLY_POSITIONS):

    prompt_last   just before the reply is written: is the model about to lie?
    reply_last    the reply's last token: the finished reply
    reply_mean    the mean over the reply's tokens

Each cell is the probe's AUROC on lie vs truthful in the tested category; the
diagonal is the in-category reference, 5-fold cross-validated. A category with
fewer than training.MIN_ROWS_PER_CLASS rows in a class is not trained on, and a
cell with fewer than generalization.FEW of either is marked.
"""

from __future__ import annotations

from dyl_repro.core import training
from dyl_repro.core.prefill import REPLY_POSITIONS
from dyl_repro.datasets.followups import NO_FOLLOWUP
from dyl_repro.datasets.gigi.analyses.generalization import CATEGORIES, INDEX

POSITIONS = REPLY_POSITIONS
MEASURE = "original"  # the cell key: AUROC on Gigi's label for the reply


def key(dataset: str) -> str:
    return f"{dataset}/{NO_FOLLOWUP}"


def probe_name(dataset: str, position: str) -> str:
    return training.probe_name(key(dataset), training.DEFAULT_LAYER, training.DEFAULT_L2_LAMBDA, position)


def plan() -> list[tuple[str, list[str], str]]:
    """(train key, test keys, read position) for every probe the matrices are built from."""
    return [(key(train), [key(test) for test in CATEGORIES if test != train], position)
            for position in POSITIONS for train in CATEGORIES]


def result_paths(position: str, train: str) -> tuple[str, dict[str, str]]:
    """Hub paths of one probe's training result and of its result on each other category."""
    name = f"{probe_name(train, position)}_hs{INDEX}"
    tests = {test: f"{key(test)}/results/{name}.json" for test in CATEGORIES if test != train}
    return f"{key(train)}/results/{name}_training.json", tests


def matrix(position: str, trained: dict[str, dict], results: dict[str, dict[str, dict | None]]) -> dict:
    """One position's matrix, in generalization.matrix's shape with the cell key MEASURE.

    trained: category -> its training result; results: category -> test column -> result (None if not on the Hub)."""
    rows = []
    for train in CATEGORIES:
        was_trained = "skipped" not in trained[train]
        cells = []
        for test in CATEGORIES:
            if not was_trained:
                cells.append({"kind": "not_trained", MEASURE: None})
            elif test == train:
                cv = trained[train]["cross_validated"]
                cells.append({"kind": "in_category", MEASURE: {k: cv[k] for k in ("auroc", "n_lie", "n_honest")}})
            elif results[train][test] is None:
                cells.append({"kind": "missing", MEASURE: None})
            else:
                o = results[train][test]["original_answer"]
                cells.append({"kind": "transfer", MEASURE: {"auroc": o["auroc"], "n_lie": o["n_lie"], "n_honest": o["n_truthful"]}})
        rows.append({"train": train, "trained": was_trained, "n_lie": trained[train]["n_lie"],
                     "n_honest": trained[train]["n_honest"], "cells": cells})
    return {"position": position, "columns": list(CATEGORIES), "rows": rows}

