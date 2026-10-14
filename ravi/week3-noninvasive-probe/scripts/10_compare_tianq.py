#!/usr/bin/env python3
"""Score the generic probe the way Tianq scored his no-follow-up matrix.

Train only on instructed pairs (or Simple Deception with --recipe truth).
Score Gigi's 11 categories at prompt_last / reply_last / reply_mean, layer 58,
with his λ=0.01 logistic. Every category is transfer: none of them is in the fit.

His in-category numbers (0.71 / 0.73 / 0.80) are probes trained on that category.
The number to set beside his is the transfer mean, 0.50. See COMPARE_TIANQ.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

WEEK3 = Path(__file__).resolve().parents[1]
REPO = WEEK3.parents[1]
WEEK1 = WEEK3.parent / "week1-dyl-vs-oa"
TIANQ_REPRO = REPO / "tianq" / "dyl-probe-repro"
sys.path.insert(0, str(WEEK3 / "src"))

from noninvasive_probe.paths import bootstrap

bootstrap()
# Tianq's loaders (alibi, MASK belief splits). Must beat week 1's older copy.
sys.path.insert(0, str(TIANQ_REPRO))

from dyl_vs_oa.metrics import summarize, write_json
from noninvasive_probe.categories import CATEGORIES, FEW, allow_dataset_import, load_category
from noninvasive_probe.config import load_config
from noninvasive_probe.data_intent import load_instructed_pairs
from noninvasive_probe.sets import assert_generic_train
from noninvasive_probe.tianq_read import (
    LAYER,
    POSITIONS,
    TIANQ_NO_FOLLOWUP,
    aisi_logits,
    fit_aisi_logistic,
    hidden_states_tianq_positions,
)

CACHE = WEEK3 / "results" / "tianq_compare_cache"


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


def _train_rows(cfg, recipe: str, max_train: int):
    from dyl_vs_oa.data import load_simple_deception, split_balanced

    if recipe == "truth":
        examples = load_simple_deception(cfg, max_n=max_train + 80)
        train, _ = split_balanced(examples, eval_n=min(80, max(20, max_train // 10)))
        train = train[:max_train]
    elif recipe == "intent":
        pairs = load_instructed_pairs(cfg, max_n=max_train)
        train = pairs[:max_train] if max_train % 2 == 0 else pairs[: max_train - 1]
    else:
        raise SystemExit(f"--recipe must be intent or truth, got {recipe!r}")
    assert_generic_train(train)
    return train


def _cache_path(tag: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{tag}.npz"


def _extract(subject, rows, tag: str, max_length: int, batch_size: int, refresh: bool):
    path = _cache_path(tag)
    uids = np.array([uid for _m, _y, uid in rows])
    if path.exists() and not refresh:
        saved = np.load(path, allow_pickle=False)
        same_rows = [str(uid) for uid in saved["uids"]] == [str(uid) for uid in uids]
        if same_rows and int(saved["max_length"]) == max_length:
            acts = {name: saved[name] for name in POSITIONS}
            print(f"  cache {path.name} ({len(uids)} rows)", flush=True)
            return acts, int(saved["n_lost_prompt"])
    convos = [row[0] for row in rows]
    acts, n_lost = hidden_states_tianq_positions(
        subject, convos, layer=LAYER, max_length=max_length, batch_size=batch_size
    )
    np.savez_compressed(
        path,
        uids=uids,
        max_length=np.int64(max_length),
        n_lost_prompt=np.int64(n_lost),
        **acts,
    )
    return acts, n_lost


def _score_position(y: np.ndarray, scores: np.ndarray) -> dict:
    keep = np.isfinite(scores)
    y_k, s_k = y[keep], scores[keep]
    out = summarize(y_k, s_k)
    out["n_dropped_truncated"] = float((~keep).sum())
    n_lie = int((y_k == 1).sum())
    n_honest = int((y_k == 0).sum())
    out["few"] = n_lie < FEW or n_honest < FEW
    return out


def _mean_auc(rows: list[dict], position: str) -> float | None:
    vals = [row[position]["auc"] for row in rows if not row[position]["few"] and np.isfinite(row[position]["auc"])]
    if not vals:
        return None
    return float(np.mean(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("intent", "truth"), default="intent")
    parser.add_argument("--categories", default=",".join(CATEGORIES), help="comma list, or 'all'")
    parser.add_argument("--max-train", type=int, default=612)
    parser.add_argument("--max-per-dataset", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--counts-only", action="store_true", help="print row counts, do not load Gemma")
    parser.add_argument("--refresh", action="store_true", help="ignore activation caches")
    args = parser.parse_args()
    _load_env()

    allow_dataset_import()
    names = list(CATEGORIES) if args.categories == "all" else [n.strip() for n in args.categories.split(",") if n.strip()]
    unknown = [n for n in names if n not in CATEGORIES]
    if unknown:
        raise SystemExit(f"unknown categories {unknown}; known: {', '.join(CATEGORIES)}")

    loaded = {name: load_category(name, args.max_per_dataset) for name in names}
    print(f"{'category':<24} {'n':>5} {'lies':>5} {'truth':>5}", flush=True)
    for name, rows in loaded.items():
        y = np.array([label for _m, label, _u in rows])
        print(f"{name:<24} {len(rows):5d} {int((y == 1).sum()):5d} {int((y == 0).sum()):5d}", flush=True)
    if args.counts_only:
        return

    import dyl_repro

    if "tianq" not in Path(dyl_repro.__file__).as_posix():
        raise SystemExit(f"loaded the week-1 dyl_repro copy ({dyl_repro.__file__}); expected tianq/")

    cfg = load_config()
    train = _train_rows(cfg, args.recipe, args.max_train)
    y_train = np.array([ex.label for ex in train], dtype=np.int64)
    n_tag = "all" if args.max_per_dataset is None else str(args.max_per_dataset)
    train_tag = f"{args.recipe}_train_n{len(train)}_L{args.max_length}"

    from dyl_vs_oa.model import load_subject

    print(f"loading {cfg.model_name}", flush=True)
    subject = load_subject(cfg.model_name)
    print(f"extracting train ({args.recipe}, {len(train)} rows)", flush=True)
    train_acts, train_lost = _extract(subject, [(ex.messages, ex.label, str(i)) for i, ex in enumerate(train)], train_tag, args.max_length, args.batch_size, args.refresh)
    if train_lost:
        print(f"  train lost prompt_last on {train_lost} rows (raise --max-length if this is not zero)", flush=True)

    weights = {}
    for position in POSITIONS:
        x = train_acts[position]
        keep = np.isfinite(x).all(axis=1)
        if not keep.all():
            print(f"  fitting {position} on {int(keep.sum())}/{len(keep)} rows", flush=True)
        weights[position] = fit_aisi_logistic(x[keep], y_train[keep])
        print(f"  fit {position} layer {LAYER} λ=0.01", flush=True)

    scored = []
    for name, rows in loaded.items():
        print(f"scoring {name}", flush=True)
        acts, n_lost = _extract(
            subject,
            rows,
            f"{name}_n{n_tag}_L{args.max_length}",
            args.max_length,
            args.batch_size,
            args.refresh,
        )
        y = np.array([label for _m, label, _u in rows], dtype=np.int64)
        cell = {"category": name, "label": CATEGORIES[name], "n_lost_prompt": n_lost}
        for position in POSITIONS:
            scores = aisi_logits(acts[position], *weights[position])
            cell[position] = _score_position(y, scores)
        scored.append(cell)
        bits = "  ".join(
            f"{position}={cell[position]['auc']:.3f}" + ("*" if cell[position]["few"] else "")
            for position in POSITIONS
        )
        print(f"  {bits}", flush=True)

    means = {position: _mean_auc(scored, position) for position in POSITIONS}
    payload = {
        "recipe": args.recipe,
        "layer": LAYER,
        "l2_lambda": 0.01,
        "train_n": len(train),
        "train_sources": sorted({ex.meta.get("source") for ex in train}),
        "max_per_dataset": args.max_per_dataset,
        "tianq_no_followup": TIANQ_NO_FOLLOWUP,
        "note": (
            "Compare mean_auc to tianq transfer (0.50), not to his in-category "
            "0.71/0.73/0.80. Those diagonals are probes trained on that category. "
            "Cells with fewer than 10 lies or 10 truthful replies are marked few and left out of the mean."
        ),
        "mean_auc": means,
        "categories": scored,
    }
    out = WEEK3 / "results" / f"tianq_compare_{args.recipe}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, payload)

    print("\nmean AUROC (categories with ≥10 each class)", flush=True)
    print(f"{'position':<14} {'ours':>8} {'tianq transfer':>16} {'tianq in-category':>20}", flush=True)
    for position in POSITIONS:
        ours = means[position]
        ref = TIANQ_NO_FOLLOWUP[position]
        ours_s = "  n/a" if ours is None else f"{ours:8.3f}"
        print(f"{position:<14} {ours_s} {ref['transfer']:16.2f} {ref['in_category']:20.2f}", flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
