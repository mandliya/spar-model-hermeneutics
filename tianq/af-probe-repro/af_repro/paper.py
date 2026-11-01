"""What the paper ran for each model, read off its own code and committed results (vendor/probe-af), and
what is left to run.

    python -m af_repro.paper missing <model> <features.h5>   # its tasks not yet extracted into that file
    python -m af_repro.paper train <model>                   # its probes' training tasks
    python -m af_repro.paper test <model>                    # their test tasks
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import h5py
import pandas as pd

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "probe-af"
# The paper's layers (vendor README, step 1; scripts/train_test_probes_af.sh --layer-idx).
LAYERS = (33, 37, 40)


def default_tasks() -> list[str]:
    """scripts/extract_activation.py's DEFAULT_TASKS: every training set and AF benchmark of the paper."""
    tree = ast.parse((VENDOR / "scripts" / "extract_activation.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "DEFAULT_TASKS":
            return ast.literal_eval(node.value)
    raise SystemExit("DEFAULT_TASKS not found in the vendored extract_activation.py")


def committed(model: str) -> pd.DataFrame:
    return pd.read_csv(VENDOR / "results" / f"results_{model}_all_layers.csv")


def train_tasks(model: str) -> list[str]:
    """The tasks the paper trained this model's probes on, as its committed results list them. The plain
    parent has no AF rollouts (it never fakes), so the paper ran it on the truthfulness sets only."""
    return list(dict.fromkeys(committed(model)["train_dataset"]))


def test_tasks(model: str) -> list[str]:
    """The tasks it tested them on, held out (its CV rows score the training tasks themselves)."""
    rows = committed(model)
    return list(dict.fromkeys(rows.loc[~rows["is_cv_result"], "test_dataset"]))


def tasks(model: str) -> list[str]:
    """Every task the paper extracted for this model: the extraction tasks its results mention.
    combined__FLEED is the claims and ethics sets together; they appear as its per-set CV rows."""
    rows = committed(model)
    wanted = set(rows["train_dataset"]) | set(rows["test_dataset"])
    return [t for t in default_tasks() if t in wanted]


def missing(model: str, features: Path) -> list[str]:
    """This model's tasks with no data yet at the last layer. The extractor writes the layers in order, so a
    task present at the last one is complete; anything else is (re)extracted."""
    if not features.exists():
        return tasks(model)
    with h5py.File(features, "r") as f:
        done = set(f[f"layer_{LAYERS[-1]}"]) if f"layer_{LAYERS[-1]}" in f else set()
    return [t for t in tasks(model) if t not in done]


if __name__ == "__main__":
    command, model = sys.argv[1:3]
    if command == "missing":
        print(" ".join(missing(model, Path(sys.argv[3]))))
    elif command in ("train", "test"):
        print(" ".join(train_tasks(model) if command == "train" else test_tasks(model)))
    else:
        raise SystemExit("usage: python -m af_repro.paper <missing|train|test> <model> [features.h5]")
