#!/usr/bin/env python3
"""HTC on stored denials, then invasive control (follow-up in context).

Uses original-turn caches from 08. Probe still frozen. HTC does not need new
GPU if caches + probes exist. Invasive extract does.

See NOTES.md §1.
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
sys.path.insert(0, str(WEEK1 / "src"))
sys.path.insert(0, str(WEEK1 / "dyl-probe-repro"))

from dyl_vs_oa.metrics import summarize, write_json
from dyl_vs_oa.model import load_subject
from dyl_vs_oa.probes import load_bundle
from noninvasive_probe.config import load_config
from noninvasive_probe.extract import extract_last_token_activations
from noninvasive_probe.protocol import admitted_for_htc, htc_mask, prefill_no_texts, prompt_last_texts
from noninvasive_probe.sets import EVAL_SETS, load_eval_set

DEFAULT_DATASETS = (
    "ascii_sandbagging,sycophancy_answer,sycophancy_feedback,"
    "mask_factual,capture_the_secret,oa_mbpp"
)
HTC_CONDITIONS = ("named", "neutral")
INVASIVE_KINDS = ("dyl_native", "named", "goldowsky_dill_no")


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


def _score(bundle, acts, y) -> dict:
    out = {"n_kept": len(bundle.layers), "mean_kept": summarize(y, bundle.score(acts))}
    if 58 in bundle.probes and 58 in acts:
        out["layer_58"] = summarize(y, bundle.probes[58].predict_proba(acts[58]))
    return out


def _load_eval_cache(cache_dir: Path, dataset: str, position: str):
    npz = cache_dir / f"{dataset}__{position}.npz"
    if not npz.exists():
        return None
    data = np.load(npz)
    y = data["y"]
    acts = {int(k.split("_", 1)[1]): data[k] for k in data.files if k.startswith("layer_")}
    return acts, y


def _load_bundles(probe_dir: Path) -> dict:
    out = {}
    if not probe_dir.exists():
        return out
    for path in sorted(probe_dir.glob("*.json")):
        out[path.stem] = load_bundle(path)
    return out


def main() -> None:
    _load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--datasets", default=DEFAULT_DATASETS)
    p.add_argument("--max-per-dataset", type=int, default=None)
    p.add_argument("--position", default="mean_excl_last_5", help="original-turn pool for HTC scores")
    p.add_argument("--skip-invasive", action="store_true")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=2)
    args = p.parse_args()

    cfg = load_config(args.config)
    names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    unknown = [n for n in names if n not in EVAL_SETS]
    if unknown:
        raise SystemExit(f"unknown datasets: {unknown}")

    probe_dir = cfg.results_dir / "noninvasive_probes"
    cache_dir = cfg.results_dir / "noninvasive_cache"
    bundles = _load_bundles(probe_dir)
    if not bundles:
        raise SystemExit(f"no probes in {probe_dir}; run scripts/08_train_noninvasive_probe.py first")

    htc_path = cfg.results_dir / "noninvasive_htc.json"
    inv_path = cfg.results_dir / "noninvasive_invasive.json"
    htc_payload = {"position": args.position, "datasets": {}}
    inv_payload = {"datasets": {}}

    for name in names:
        convs = load_eval_set(name, cfg, max_n=args.max_per_dataset)
        cached = _load_eval_cache(cache_dir, name, args.position)
        if cached is None:
            print(f"{name}: missing cache {args.position}; skip HTC", flush=True)
            continue
        acts, y = cached
        if len(y) != len(convs):
            print(f"{name}: cache n={len(y)} vs convs {len(convs)}; skip HTC", flush=True)
            continue
        htc_payload["datasets"][name] = {}
        conditions = ("native",) if name in {"dyl_city", "dyl_varied", "oa_mbpp"} else HTC_CONDITIONS
        for cond in conditions:
            admitted = admitted_for_htc(name, convs, cond)
            keep, counts = htc_mask(y, admitted)
            cell = {"counts": counts, "probes": {}}
            if counts["n_lied_denied"] == 0 or counts["n_truthful_denied"] == 0:
                cell["note"] = "HTC undefined (one class empty) — OA often has almost no denials"
                htc_payload["datasets"][name][cond] = cell
                continue
            idx = np.array(keep)
            y_h = y[idx]
            acts_h = {layer: arr[idx] for layer, arr in acts.items()}
            for key, bundle in bundles.items():
                parts = key.split("__")
                if len(parts) != 3 or parts[1] != args.position:
                    continue
                cell["probes"][key] = _score(bundle, acts_h, y_h)
            htc_payload["datasets"][name][cond] = cell
        write_json(htc_path, htc_payload)
        print(f"{name}: HTC -> {htc_path}", flush=True)

    if args.skip_invasive:
        print("skipping invasive control", flush=True)
        return

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise SystemExit("HF_TOKEN is not set (needed for invasive extract).")

    subject = load_subject(
        cfg.model_name,
        dtype=cfg.dtype,
        device_map=cfg.device_map,
        attn_implementation=cfg.attn_implementation,
    )
    preferred = [k for k in bundles if k.endswith("__logistic")]
    use_keys = preferred or list(bundles)

    for name in names:
        convs = load_eval_set(name, cfg, max_n=args.max_per_dataset)
        y = np.array([c.label for c in convs], dtype=np.int64)
        inv_payload["datasets"][name] = {}
        for kind in INVASIVE_KINDS:
            if kind == "named" and name in {"dyl_city", "dyl_varied"}:
                continue
            texts_pl = prompt_last_texts(subject.tokenizer, convs, name, kind)
            texts_no = prefill_no_texts(subject.tokenizer, convs, name, kind)
            acts_pl = extract_last_token_activations(
                subject, texts_pl, layers=None, batch_size=args.batch_size, max_length=args.max_length
            )
            acts_no = extract_last_token_activations(
                subject, texts_no, layers=None, batch_size=args.batch_size, max_length=args.max_length
            )
            kind_out = {"prompt_last": {}, "prefill_no": {}}
            for key in use_keys:
                bundle = bundles[key]
                kind_out["prompt_last"][key] = _score(bundle, acts_pl, y)
                kind_out["prefill_no"][key] = _score(bundle, acts_no, y)
            inv_payload["datasets"][name][kind] = kind_out
            write_json(inv_path, inv_payload)
            print(f"{name} {kind}: invasive -> {inv_path}", flush=True)

    print(f"Wrote {htc_path} and {inv_path}", flush=True)


if __name__ == "__main__":
    main()
