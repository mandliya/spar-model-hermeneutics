#!/usr/bin/env python3
"""Train noninvasive probes (no follow-up) at each original-turn pool; score eval sets.

Recipes:
  truth   Simple Deception facts as the assistant turn (Geometry of Truth)
  intent  instructed honest vs lie system prompt, same statement (Goldowsky-Dill)

Heads: logistic (OA-style), mean_diff, and LAT on intent pairs.

Does not train on Gigi or OA honeypot. See NOTES.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

WEEK3 = Path(__file__).resolve().parents[1]
WEEK1 = WEEK3.parent / "week1-dyl-vs-oa"
sys.path.insert(0, str(WEEK3 / "src"))
sys.path.insert(0, str(WEEK1 / "src"))
sys.path.insert(0, str(WEEK1 / "dyl-probe-repro"))

from dyl_vs_oa.data import load_simple_deception, split_balanced
from dyl_vs_oa.metrics import summarize, write_json
from dyl_vs_oa.model import load_subject
from dyl_vs_oa.probes import load_bundle, save_bundle, train_oa_bundle
from noninvasive_probe.config import load_config
from noninvasive_probe.data_intent import load_instructed_pairs
from noninvasive_probe.extract import ORIGINAL_TURN_POSITIONS, extract_original_turn_activations_at_positions
from noninvasive_probe.heads import train_lat_bundle, train_mean_diff_bundle
from noninvasive_probe.protocol import comparison_row
from noninvasive_probe.sets import EVAL_SETS, load_eval_set

DEFAULT_DATASETS = (
    "ascii_sandbagging,sycophancy_answer,sycophancy_feedback,"
    "mask_factual,capture_the_secret,oa_mbpp"
)


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


def _score(bundle, acts: dict[int, np.ndarray], y: np.ndarray) -> dict:
    out = {"n_kept": len(bundle.layers), "mean_kept": summarize(y, bundle.score(acts))}
    if 58 in bundle.probes and 58 in acts:
        out["layer_58"] = summarize(y, bundle.probes[58].predict_proba(acts[58]))
    return out


def _train_examples(cfg, recipe: str, max_train: int):
    if recipe == "truth":
        examples = load_simple_deception(cfg, max_n=max_train + 80)
        train, _ = split_balanced(examples, eval_n=min(80, max(20, max_train // 10)))
        return train[:max_train]
    if recipe == "intent":
        pairs = load_instructed_pairs(cfg, max_n=max_train)
        return pairs[:max_train] if max_train % 2 == 0 else pairs[: max_train - 1]
    raise KeyError(recipe)


def _fit(recipe, position, head, acts, y, cfg, model_name):
    kwargs = dict(
        followup_user="",
        followup_assistant=None,
        model_name=model_name,
        extraction="original_turn",
    )
    if head == "logistic":
        bundle = train_oa_bundle(
            acts,
            y,
            C=cfg.oa.C,
            ce_keep_below=cfg.oa.ce_keep_below,
            followup_user="",
            followup_assistant="",
            model_name=model_name,
            standardize=True,
            solver="liblinear",
        )
        return replace(
            bundle,
            method=f"noninvasive_{recipe}",
            extraction="original_turn",
            followup_assistant=None,
        )
    if head == "mean_diff":
        return train_mean_diff_bundle(acts, y, method=f"mean_diff_{recipe}", **kwargs)
    if head == "lat":
        if recipe != "intent":
            raise SystemExit("LAT is only defined on instructed pairs (recipe=intent)")
        return train_lat_bundle(acts, y, **kwargs)
    raise KeyError(head)


def _save_eval_cache(cache_dir: Path, dataset: str, position: str, acts, y, uids) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {f"layer_{layer}": arr for layer, arr in acts.items()}
    payload["y"] = y
    np.savez_compressed(cache_dir / f"{dataset}__{position}.npz", **payload)
    (cache_dir / f"{dataset}__{position}.meta.json").write_text(
        json.dumps({"n": int(len(y)), "uids": uids, "layers": sorted(acts)})
    )


def _load_eval_cache(cache_dir: Path, dataset: str, position: str):
    npz = cache_dir / f"{dataset}__{position}.npz"
    meta_path = cache_dir / f"{dataset}__{position}.meta.json"
    if not npz.exists():
        return None
    data = np.load(npz)
    y = data["y"]
    acts = {int(k.split("_", 1)[1]): data[k] for k in data.files if k.startswith("layer_")}
    uids = json.loads(meta_path.read_text())["uids"] if meta_path.exists() else [str(i) for i in range(len(y))]
    return acts, y, uids


def main() -> None:
    _load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--max-train", type=int, default=400)
    p.add_argument("--recipes", default="truth,intent")
    p.add_argument("--heads", default="logistic,mean_diff")
    p.add_argument("--positions", default=",".join(ORIGINAL_TURN_POSITIONS))
    p.add_argument("--datasets", default=DEFAULT_DATASETS)
    p.add_argument("--max-per-dataset", type=int, default=None)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--reuse-probes", action="store_true")
    args = p.parse_args()

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise SystemExit("HF_TOKEN is not set (Gemma-3 is gated).")

    cfg = load_config(args.config)
    recipes = [x.strip() for x in args.recipes.split(",") if x.strip()]
    heads = [x.strip() for x in args.heads.split(",") if x.strip()]
    positions = tuple(x.strip() for x in args.positions.split(",") if x.strip())
    names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    unknown = [n for n in names if n not in EVAL_SETS]
    if unknown:
        raise SystemExit(f"unknown datasets: {unknown}")

    probe_dir = cfg.results_dir / "noninvasive_probes"
    cache_dir = cfg.results_dir / "noninvasive_cache"
    out_path = cfg.results_dir / "noninvasive_eval.json"
    probe_dir.mkdir(parents=True, exist_ok=True)

    subject = load_subject(
        cfg.model_name,
        dtype=cfg.dtype,
        device_map=cfg.device_map,
        attn_implementation=cfg.attn_implementation,
    )
    print(f"Model {subject.name}: {subject.n_layers} layers, d={subject.hidden_size}", flush=True)

    bundles = {}
    for recipe in recipes:
        train = _train_examples(cfg, recipe, args.max_train)
        y_train = np.array([ex.label for ex in train], dtype=np.int64)
        print(f"train {recipe}: {len(train)} rows", flush=True)
        train_by_pos = None
        for position in positions:
            for head in heads:
                if head == "lat" and recipe != "intent":
                    continue
                key = f"{recipe}__{position}__{head}"
                path = probe_dir / f"{key}.json"
                if args.reuse_probes and path.exists():
                    bundles[key] = load_bundle(path)
                    print(f"loaded {path}", flush=True)
                    continue
                if train_by_pos is None:
                    train_by_pos = extract_original_turn_activations_at_positions(
                        subject,
                        train,
                        layers=None,
                        batch_size=args.batch_size,
                        max_length=args.max_length,
                        positions=positions,
                    )
                bundle = _fit(recipe, position, head, train_by_pos[position], y_train, cfg, cfg.model_name)
                save_bundle(bundle, path)
                bundles[key] = bundle
                print(f"wrote {path} kept={len(bundle.layers)}", flush=True)
        train_by_pos = None

    payload = {
        "protocol": "original_turn_no_followup",
        "positions": list(positions),
        "recipes": recipes,
        "heads": heads,
        "datasets": {},
        "comparison": {},
    }
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        payload["datasets"] = dict(prev.get("datasets") or {})
        payload["comparison"] = dict(prev.get("comparison") or {})

    for name in names:
        convs = load_eval_set(name, cfg, max_n=args.max_per_dataset)
        y = np.array([c.label for c in convs], dtype=np.int64)
        uids = [str(c.meta.get("uid", i)) for i, c in enumerate(convs)]
        missing = [pos for pos in positions if _load_eval_cache(cache_dir, name, pos) is None]
        if missing:
            print(f"{name}: extracting {len(convs)} at {missing}", flush=True)
            extracted = extract_original_turn_activations_at_positions(
                subject,
                convs,
                layers=None,
                batch_size=args.batch_size,
                max_length=args.max_length,
                positions=tuple(missing) if set(missing) != set(positions) else positions,
            )
            for pos, acts in extracted.items():
                _save_eval_cache(cache_dir, name, pos, acts, y, uids)

        payload["datasets"].setdefault(name, {"n": int(len(y)), "positions": {}})
        known = comparison_row(name)
        for position in positions:
            cached = _load_eval_cache(cache_dir, name, position)
            acts, y_cached, _uids = cached
            pos_out = {}
            for key, bundle in bundles.items():
                _recipe, pos, _head = key.split("__")
                if pos != position:
                    continue
                pos_out[key] = _score(bundle, acts, y_cached)
            payload["datasets"][name]["positions"][position] = pos_out
        payload["comparison"][name] = {
            "just_asking": known.get("just_asking"),
            "dyl_probe": known.get("dyl_probe"),
            "oa_layer_58": known.get("oa_layer_58"),
            "oa_mean": known.get("oa_mean"),
            "n_lie": int((y == 1).sum()),
            "n_honest": int((y == 0).sum()),
            "note": known.get("note"),
        }
        write_json(out_path, payload)
        print(f"{name}: wrote scores -> {out_path}", flush=True)

    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
