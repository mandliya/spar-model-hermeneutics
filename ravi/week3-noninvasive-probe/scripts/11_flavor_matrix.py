#!/usr/bin/env python3
"""Train one non-invasive probe per deception flavor and test it on the others.

No follow-up is in the forward pass. The diagonal is 5-fold out-of-fold AUROC.
This does not train on instructed pairs. See WEEK4_PLAN.md steps 1–6.

    python -u scripts/11_flavor_matrix.py --counts-only
    python -u scripts/11_flavor_matrix.py --extract-only
    python -u scripts/11_flavor_matrix.py --fit-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

WEEK3 = Path(__file__).resolve().parents[1]
WEEK1 = WEEK3.parent / "week1-dyl-vs-oa"
sys.path.insert(0, str(WEEK3 / "src"))

from noninvasive_probe.categories import CATEGORIES, TRAIN_FLAVORS, allow_dataset_import, load_category, resolve_flavors
from noninvasive_probe.flavor_matrix import (
    LAYER,
    TIANQ_POSITIONS,
    cache_path,
    ensure_cached,
    family_of,
    matrix_for_position,
    render_position,
    require_flavor_cache,
    resolve_positions,
    result_signature,
    upsert_result,
)
from noninvasive_probe.paths import bootstrap

bootstrap()

CACHE = WEEK3 / "results" / "flavor_cache"
RESULT = WEEK3 / "results" / "flavor_matrix.json"


def _load_env() -> None:
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


def _print_counts(loaded: dict) -> None:
    print(f"{'category':<24} {'n':>5} {'lies':>5} {'truth':>5}", flush=True)
    for name, rows in loaded.items():
        y = np.array([label for _messages, label, _uid in rows])
        print(f"{name:<24} {len(rows):5d} {int((y == 1).sum()):5d} {int((y == 0).sum()):5d}", flush=True)


def _pieces(rows):
    messages = [messages for messages, _label, _uid in rows]
    uids = [uid for _messages, _label, uid in rows]
    y = np.array([label for _messages, label, _uid in rows], dtype=np.int64)
    return messages, uids, y


def _extract_flavor(subject, flavor: str, rows, positions: list[str], max_length: int, batch_size: int, refresh: bool) -> None:
    messages, uids, y = _pieces(rows)
    closed = [position for position in positions if family_of(position) == "closed"]
    tianq = [position for position in positions if family_of(position) == "tianq"]
    if closed:
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

        path = cache_path(CACHE, flavor, "closed", max_length)
        ensure_cached(path, uids, y, max_length, closed, refresh=refresh, extract_missing=extract_closed)
        print(f"  cached {path.name}", flush=True)
    if tianq:
        def extract_tianq(_missing: list[str]):
            from noninvasive_probe.tianq_read import hidden_states_tianq_positions

            acts, n_lost = hidden_states_tianq_positions(
                subject, messages, layer=LAYER, max_length=max_length, batch_size=batch_size
            )
            # One forward pass produces all three reads. Keep them all.
            return {position: acts[position] for position in TIANQ_POSITIONS}, n_lost

        path = cache_path(CACHE, flavor, "tianq", max_length)
        saved = ensure_cached(path, uids, y, max_length, tianq, refresh=refresh, extract_missing=extract_tianq)
        print(f"  cached {path.name} (prompt tokens lost: {saved['n_lost_prompt']})", flush=True)


def _fit(flavors: list[str], loaded: dict, positions: list[str], max_length: int, max_per_dataset: int | None) -> None:
    signature = result_signature(flavors, positions, max_length, max_per_dataset)
    train = [name for name in TRAIN_FLAVORS if name in flavors]
    for position in positions:
        by_flavor = {}
        for flavor, rows in loaded.items():
            _messages, uids, y = _pieces(rows)
            path = cache_path(CACHE, flavor, family_of(position), max_length)
            saved = require_flavor_cache(path, uids, y, max_length, [position])
            if saved["n_lost_prompt"]:
                note = f"{flavor}: lost prompt_last on {saved['n_lost_prompt']} rows."
                if position == "prompt_last":
                    raise SystemExit(note + " Delete results/flavor_cache and raise --max-length.")
                print(f"  warning: {note}", flush=True)
            by_flavor[flavor] = (saved["acts"][position], saved["y"])
        matrix = matrix_for_position(by_flavor, position=position, train_flavors=train, test_flavors=flavors)
        markdown = render_position(matrix, CATEGORIES)
        print("\n" + markdown + "\n", flush=True)
        upsert_result(RESULT, signature, matrix, markdown)
        print(f"wrote {position} -> {RESULT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--flavors", default="all")
    parser.add_argument("--positions", default="all")
    parser.add_argument("--max-per-dataset", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    modes = (args.counts_only, args.extract_only, args.fit_only)
    if sum(bool(mode) for mode in modes) != 1:
        raise SystemExit("pass exactly one of --counts-only, --extract-only, --fit-only")

    flavors = resolve_flavors(args.flavors)
    positions = resolve_positions(args.positions)
    _load_env()
    allow_dataset_import()
    loaded = {name: load_category(name, args.max_per_dataset) for name in flavors}
    _print_counts(loaded)
    if args.counts_only:
        return

    if args.extract_only:
        _load_env()
        from noninvasive_probe.config import load_config
        from dyl_vs_oa.model import load_subject

        cfg = load_config()
        print(f"loading {cfg.model_name}", flush=True)
        subject = load_subject(cfg.model_name)
        for flavor, rows in loaded.items():
            print(f"extracting {flavor} ({len(rows)} rows)", flush=True)
            _extract_flavor(subject, flavor, rows, positions, args.max_length, args.batch_size, args.refresh)
        return

    _fit(flavors, loaded, positions, args.max_length, args.max_per_dataset)


if __name__ == "__main__":
    main()
