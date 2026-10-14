"""Load, extract, and fit one flavor matrix. Shared by the CLI and the week 5 runner.

The runner logs each return value before starting the next flavor or position.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from noninvasive_probe.categories import TRAIN_FLAVORS
from noninvasive_probe.flavor_matrix import (
    LAYER,
    TIANQ_POSITIONS,
    cache_path,
    ensure_cached,
    family_of,
    matrix_for_position,
    read_cache,
    render_position,
    require_flavor_cache,
    result_signature,
    same_rows,
    upsert_result,
)
from noninvasive_probe.paths import WEEK3

WEEK1 = WEEK3.parent / "week1-dyl-vs-oa"


def load_env() -> None:
    for path in (
        WEEK1 / "dyl-probe-repro" / ".env",
        WEEK3 / ".env",
        WEEK1 / ".env",
        Path("/workspace/.env"),
    ):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def pieces(rows):
    messages = [messages for messages, _label, _uid in rows]
    uids = [uid for _messages, _label, uid in rows]
    y = np.array([label for _messages, label, _uid in rows], dtype=np.int64)
    return messages, uids, y


def observed_counts(loaded: dict) -> dict[str, tuple[int, int, int]]:
    counts = {}
    for name, rows in loaded.items():
        _messages, _uids, y = pieces(rows)
        counts[name] = (len(rows), int((y == 1).sum()), int((y == 0).sum()))
    return counts


def _family_detail(saved: dict, family: str, path: Path, positions: list[str], reused: bool) -> dict:
    described = {}
    for position in positions:
        arr = np.asarray(saved["acts"][position])
        finite = int(np.isfinite(arr).all(axis=1).sum()) if arr.ndim == 2 else int(np.isfinite(arr).sum())
        described[position] = {"shape": list(arr.shape), "finite": finite, "rows": int(arr.shape[0])}
    return {
        "family": family,
        "path": str(path),
        "reused": reused,
        "n_lost_prompt": int(saved["n_lost_prompt"]),
        "positions": described,
    }


def extract_flavor(
    subject,
    flavor: str,
    rows,
    positions: list[str],
    max_length: int,
    batch_size: int,
    refresh: bool,
    cache_dir: Path,
) -> dict:
    """Extract this flavor. Returns a detail dict; does not print."""
    messages, uids, y = pieces(rows)
    started = time.perf_counter()
    families = []
    closed = [position for position in positions if family_of(position) == "closed"]
    tianq = [position for position in positions if family_of(position) == "tianq"]
    if closed:
        path = cache_path(cache_dir, flavor, "closed", max_length)
        existing = None if refresh else read_cache(path)
        reused = existing is not None and same_rows(existing, uids, y, max_length) and all(
            position in existing["acts"] for position in closed
        )

        def extract_closed(missing: list[str]):
            from dyl_vs_oa.data import LabeledConversation
            from noninvasive_probe.extract import extract_original_turn_activations_at_positions

            convos = [
                LabeledConversation(message, int(label), {"uid": uid, "source": flavor})
                for message, label, uid in zip(messages, y, uids)
            ]
            extracted = extract_original_turn_activations_at_positions(
                subject,
                convos,
                layers=[LAYER],
                batch_size=batch_size,
                max_length=max_length,
                positions=tuple(missing),
            )
            return {position: extracted[position][LAYER] for position in missing}, 0

        saved = ensure_cached(path, uids, y, max_length, closed, refresh=refresh, extract_missing=extract_closed)
        families.append(_family_detail(saved, "closed", path, closed, reused))
    if tianq:
        path = cache_path(cache_dir, flavor, "tianq", max_length)
        existing = None if refresh else read_cache(path)
        reused = existing is not None and same_rows(existing, uids, y, max_length) and all(
            position in existing["acts"] for position in tianq
        )

        def extract_tianq(_missing: list[str]):
            from noninvasive_probe.tianq_read import hidden_states_tianq_positions

            acts, n_lost = hidden_states_tianq_positions(
                subject, messages, layer=LAYER, max_length=max_length, batch_size=batch_size
            )
            return {position: acts[position] for position in TIANQ_POSITIONS}, n_lost

        saved = ensure_cached(path, uids, y, max_length, tianq, refresh=refresh, extract_missing=extract_tianq)
        families.append(_family_detail(saved, "tianq", path, list(TIANQ_POSITIONS), reused))
    n_lie, n_honest = int((y == 1).sum()), int((y == 0).sum())
    return {
        "flavor": flavor,
        "n": len(rows),
        "n_lie": n_lie,
        "n_honest": n_honest,
        "seconds": time.perf_counter() - started,
        "reused": bool(families) and all(family["reused"] for family in families),
        "families": families,
    }


def fit_position(
    flavor_names: list[str],
    loaded: dict,
    position: str,
    max_length: int,
    cache_dir: Path,
    labels: dict[str, str],
) -> tuple[dict, str, list[str]]:
    """Fit one position. The caller writes the JSON so a multi-position signature stays intact."""
    train = [name for name in TRAIN_FLAVORS if name in flavor_names]
    by_flavor = {}
    warnings = []
    for flavor, rows in loaded.items():
        _messages, uids, y = pieces(rows)
        path = cache_path(cache_dir, flavor, family_of(position), max_length)
        saved = require_flavor_cache(path, uids, y, max_length, [position])
        if saved["n_lost_prompt"]:
            note = f"{flavor}: lost prompt_last on {saved['n_lost_prompt']} rows."
            if position == "prompt_last":
                raise SystemExit(note + " Delete results/flavor_cache and raise --max-length.")
            warnings.append(note)
        by_flavor[flavor] = (saved["acts"][position], saved["y"])
    matrix = matrix_for_position(by_flavor, position=position, train_flavors=train, test_flavors=flavor_names)
    markdown = render_position(matrix, labels)
    return matrix, markdown, warnings


def fit_positions(
    flavor_names: list[str],
    loaded: dict,
    positions: list[str],
    max_length: int,
    max_per_dataset: int | None,
    cache_dir: Path,
    result_path: Path,
    labels: dict[str, str],
) -> list[tuple[dict, str, list[str]]]:
    signature = result_signature(flavor_names, positions, max_length, max_per_dataset)
    written = []
    for position in positions:
        matrix, markdown, warnings = fit_position(
            flavor_names, loaded, position, max_length, cache_dir, labels
        )
        upsert_result(result_path, signature, matrix, markdown)
        written.append((matrix, markdown, warnings))
    return written
