"""Per-flavor probe matrix. Fit from cached activations; no model and no instructed pairs.

Diagonal cells are 5-fold stratified out-of-fold scores. Off-diagonal cells are one
fit on the train flavor scored on the test flavor. Both use Tianq's logistic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from noninvasive_probe.categories import FEW, FLAVOR_ORDER, TRAIN_FLAVORS
from noninvasive_probe.extract import ORIGINAL_TURN_POSITIONS
from noninvasive_probe.tianq_read import LAYER, POSITIONS, TIANQ_NO_FOLLOWUP, aisi_logits, fit_aisi_logistic

FOLDS = 5
SEED = 0
CLOSED_POSITIONS = ORIGINAL_TURN_POSITIONS
TIANQ_POSITIONS = POSITIONS
POSITION_ORDER = (
    "mean_answer",
    "reply_mean",
    "prompt_last",
    "reply_last",
    "last_content",
    "end_of_turn",
    "last_newline",
    "mean_excl_last_5",
    "mean_first_5",
)
CACHE_META = ("uids", "y", "max_length", "n_lost_prompt")


def resolve_positions(raw: str | None) -> list[str]:
    if raw is None or raw.strip() in {"", "all"}:
        return list(POSITION_ORDER)
    asked = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in asked if name not in POSITION_ORDER]
    if unknown:
        raise SystemExit(f"unknown positions {unknown}; known: {', '.join(POSITION_ORDER)}")
    return [name for name in POSITION_ORDER if name in set(asked)]


def family_of(position: str) -> str:
    if position in CLOSED_POSITIONS:
        return "closed"
    if position in TIANQ_POSITIONS:
        return "tianq"
    raise KeyError(f"unknown position {position!r}")


def cache_path(cache_dir: Path, flavor: str, family: str, max_length: int) -> Path:
    return cache_dir / f"{flavor}_{family}_L{max_length}.npz"


def class_counts(y: np.ndarray) -> tuple[int, int]:
    y = np.asarray(y)
    return int((y == 1).sum()), int((y == 0).sum())


def is_few(y: np.ndarray) -> bool:
    n_lie, n_honest = class_counts(y)
    return n_lie < FEW or n_honest < FEW


def can_train(y: np.ndarray) -> bool:
    return not is_few(y)


def stratified_folds(y: np.ndarray, folds: int = FOLDS, seed: int = SEED):
    y = np.asarray(y)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


def out_of_fold_scores(x: np.ndarray, y: np.ndarray, folds: int = FOLDS, seed: int = SEED) -> np.ndarray:
    """Score every row from a fit that did not include it."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    scores = np.full(len(y), np.nan, dtype=np.float64)
    for train_idx, test_idx in stratified_folds(y, folds=folds, seed=seed):
        if set(train_idx.tolist()).intersection(test_idx.tolist()):
            raise RuntimeError("a CV fold overlaps itself")
        weights = fit_aisi_logistic(x[train_idx], y[train_idx])
        scores[test_idx] = aisi_logits(x[test_idx], *weights)
    if not np.isfinite(scores).all():
        raise RuntimeError("a row was left out of every fold")
    return scores


def _auroc(y: np.ndarray, scores: np.ndarray) -> float | None:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def _cell(kind: str, y: np.ndarray, scores: np.ndarray | None) -> dict:
    n_lie, n_honest = class_counts(y)
    few = n_lie < FEW or n_honest < FEW
    auroc = None if scores is None else _auroc(y, scores)
    return {
        "kind": kind,
        "auroc": auroc,
        "n_lie": n_lie,
        "n_honest": n_honest,
        "few": few or auroc is None,
    }


def matrix_for_position(
    by_flavor: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    position: str,
    train_flavors: tuple[str, ...] | list[str] | None = None,
    test_flavors: tuple[str, ...] | list[str] | None = None,
) -> dict:
    """One position's train × test matrix. `by_flavor` maps a flavor to (activations, labels)."""
    present = [name for name in FLAVOR_ORDER if name in by_flavor]
    trains = [name for name in (train_flavors if train_flavors is not None else TRAIN_FLAVORS) if name in by_flavor]
    tests = [name for name in (test_flavors if test_flavors is not None else present) if name in by_flavor]
    rows = []
    for train_name in trains:
        x, y = _finite(by_flavor[train_name])
        trained = can_train(y)
        n_lie, n_honest = class_counts(y)
        cells = []
        weights = fit_aisi_logistic(x, y) if trained else None
        oof = out_of_fold_scores(x, y) if trained else None
        for test_name in tests:
            if not trained:
                xt, yt = _finite(by_flavor[test_name])
                cells.append({"test": test_name, **_cell("not_trained", yt, None)})
                continue
            if test_name == train_name:
                cells.append({"test": test_name, **_cell("in_category", y, oof)})
                continue
            xt, yt = _finite(by_flavor[test_name])
            scores = aisi_logits(xt, *weights)
            cells.append({"test": test_name, **_cell("transfer", yt, scores)})
        rows.append({"train": train_name, "trained": trained, "n_lie": n_lie, "n_honest": n_honest, "cells": cells})
    return _summarize(position, trains, tests, rows)


def _finite(pair: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x, y = pair
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if len(x) != len(y):
        raise ValueError(f"activations and labels differ in length ({len(x)} vs {len(y)})")
    keep = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return x[keep], y[keep]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _summarize(position: str, trains: list[str], tests: list[str], rows: list[dict]) -> dict:
    in_category = [
        cell["auroc"]
        for row in rows
        for cell in row["cells"]
        if cell["kind"] == "in_category" and not cell["few"] and cell["auroc"] is not None
    ]
    transfer = [
        cell["auroc"]
        for row in rows
        for cell in row["cells"]
        if cell["kind"] == "transfer" and not cell["few"] and cell["auroc"] is not None
    ]
    return {
        "position": position,
        "layer": LAYER,
        "folds": FOLDS,
        "seed": SEED,
        "train": trains,
        "test": tests,
        "mean_in_category": _mean(in_category),
        "mean_transfer": _mean(transfer),
        "transfer_cells": len(transfer),
        "transfer_cells_at_least_0_7": sum(score >= 0.7 for score in transfer),
        "rows": rows,
    }


def format_cell(cell: dict) -> str:
    if cell["kind"] == "not_trained":
        return "not trained"
    if cell["auroc"] is None:
        return "-"
    few = " (few)" if cell["few"] else ""
    mark = "**" if cell["kind"] == "in_category" else ""
    return f"{mark}{cell['auroc']:.2f}{mark} ({cell['n_lie']}/{cell['n_honest']}){few}"


def _fmt_mean(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_position(matrix: dict, labels: dict[str, str]) -> str:
    position = matrix["position"]
    line = (
        f"NO FOLLOW-UP, one probe per flavor, read at {position}: "
        f"in-category {_fmt_mean(matrix['mean_in_category'])}, "
        f"transfer {_fmt_mean(matrix['mean_transfer'])} "
        f"({matrix['transfer_cells_at_least_0_7']} of {matrix['transfer_cells']} "
        "transfer cells at 0.7 or more)."
    )
    reference = TIANQ_NO_FOLLOWUP.get(position)
    if reference is not None:
        line += (
            f"\nTianq's category-trained probes at this position: "
            f"in-category {reference['in_category']:.2f}, transfer {reference['transfer']:.2f}."
        )
    names = [labels.get(name, name) for name in matrix["test"]]
    header = "| trained on \\ tested on | " + " | ".join(names) + " |"
    rule = "|" + " --- |" * (len(names) + 1)
    body = []
    for row in matrix["rows"]:
        title = labels.get(row["train"], row["train"])
        if not row["trained"]:
            title += f" ({row['n_lie']} lies / {row['n_honest']} honest: too few)"
        body.append("| " + title + " | " + " | ".join(format_cell(cell) for cell in row["cells"]) + " |")
    return "\n".join([line, "", header, rule, *body])


def same_rows(saved: dict, uids: list[str], y: np.ndarray, max_length: int) -> bool:
    return (
        saved["uids"] == [str(uid) for uid in uids]
        and np.array_equal(saved["y"], np.asarray(y, dtype=np.int64))
        and saved["max_length"] == max_length
    )


def read_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    with np.load(path) as data:
        acts = {key: np.array(data[key]) for key in data.files if key not in CACHE_META}
        return {
            "uids": [str(uid) for uid in data["uids"]],
            "y": np.array(data["y"], dtype=np.int64),
            "max_length": int(data["max_length"]),
            "n_lost_prompt": int(data["n_lost_prompt"]) if "n_lost_prompt" in data.files else 0,
            "acts": acts,
        }


def write_cache(
    path: Path,
    uids: list[str],
    y: np.ndarray,
    max_length: int,
    acts: dict[str, np.ndarray],
    n_lost_prompt: int = 0,
) -> None:
    y = np.asarray(y, dtype=np.int64)
    for name, arr in acts.items():
        if len(arr) != len(y):
            raise ValueError(f"{name} has {len(arr)} rows, labels have {len(y)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        uids=np.array([str(uid) for uid in uids]),
        y=y,
        max_length=np.int64(max_length),
        n_lost_prompt=np.int64(n_lost_prompt),
        **{name: np.asarray(arr) for name, arr in acts.items()},
    )


def require_flavor_cache(
    path: Path,
    uids: list[str],
    y: np.ndarray,
    max_length: int,
    positions: list[str],
) -> dict:
    """Load one flavor's cache. Refuse a missing file or a shorter extract than this run asked for."""
    saved = read_cache(path)
    if saved is None:
        raise SystemExit(f"missing cache {path}. Run --extract-only before --fit-only.")
    if not same_rows(saved, uids, y, max_length):
        raise SystemExit(
            f"{path.name} has {len(saved['uids'])} rows at max_length {saved['max_length']}; "
            f"this run asked for {len(uids)} rows at max_length {max_length}. "
            "Refusing a smaller or different extract. Delete results/flavor_cache and re-extract."
        )
    missing = [position for position in positions if position not in saved["acts"]]
    if missing:
        raise SystemExit(f"{path.name} is missing {missing}. Re-extract those positions.")
    return saved


def ensure_cached(
    path: Path,
    uids: list[str],
    y: np.ndarray,
    max_length: int,
    positions: list[str],
    *,
    refresh: bool,
    extract_missing,
) -> dict:
    """Return the saved cache, extracting only positions that are not already stored for these rows.

    `extract_missing(positions) -> (acts, n_lost_prompt)`.
    """
    saved = None if refresh else read_cache(path)
    if saved is not None and not same_rows(saved, uids, y, max_length):
        saved = None
    acts = {} if saved is None else dict(saved["acts"])
    n_lost = 0 if saved is None else saved["n_lost_prompt"]
    missing = [position for position in positions if position not in acts]
    if missing or saved is None:
        new_acts, n_lost = extract_missing(missing or list(positions))
        acts.update(new_acts)
        write_cache(path, uids, y, max_length, acts, n_lost_prompt=n_lost)
        saved = read_cache(path)
    return saved


def result_signature(flavors: list[str], positions: list[str], max_length: int, max_per_dataset: int | None) -> dict:
    return {
        "flavors": list(flavors),
        "positions": list(positions),
        "max_length": max_length,
        "max_per_dataset": max_per_dataset,
        "layer": LAYER,
        "folds": FOLDS,
        "seed": SEED,
    }


def upsert_result(path: Path, signature: dict, matrix: dict, markdown: str) -> dict:
    """Keep finished positions when a later position crashes. A different signature starts over."""
    payload = {"signature": signature, "positions": {}, "markdown": {}}
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("signature") == signature:
            payload = previous
    payload["positions"][matrix["position"]] = matrix
    payload["markdown"][matrix["position"]] = markdown
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload
