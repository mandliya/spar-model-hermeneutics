#!/usr/bin/env python3
"""Train (or load) the OA Simple Deception prefill probe and score it on DYL + Gigi.

Mentor note: test OA on DYL; if there is headroom, find the difference; otherwise
ablate “did you lie” vs “did you sandbag” on Gigi / Anthropic sets.

OA already has numbers on OA MBPP-Honeypot (L58 0.920). This script is the
reverse transfer. Does not re-extract Gigi for the official DYL probe.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dyl-probe-repro"))

from dyl_vs_oa.config import load_config
from dyl_vs_oa.data import load_simple_deception, split_balanced
from dyl_vs_oa.extract import extract_prefill_activations
from dyl_vs_oa.metrics import summarize, write_json
from dyl_vs_oa.model import load_subject
from dyl_vs_oa.oa_transfer import KNOWN_DYL, TRANSFER_SETS, load_transfer_set
from dyl_vs_oa.probes import load_bundle, save_bundle, train_oa_bundle


def _load_env() -> None:
    for path in (ROOT / "dyl-probe-repro" / ".env", ROOT / ".env", Path("/workspace/.env")):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _score_bundle(bundle, acts: dict[int, np.ndarray], y: np.ndarray) -> dict:
    out = {"kept_layers": bundle.layers, "n_kept": len(bundle.layers)}
    out["mean_kept"] = summarize(y, bundle.score(acts))
    if 58 in bundle.probes and 58 in acts:
        out["layer_58"] = summarize(y, bundle.probes[58].predict_proba(acts[58]))
    return out


def _cache_paths(cache_dir: Path, name: str) -> tuple[Path, Path]:
    return cache_dir / f"{name}.npz", cache_dir / f"{name}.meta.json"


def _save_acts(cache_dir: Path, name: str, acts: dict[int, np.ndarray], y: np.ndarray, n_truncated: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz, meta = _cache_paths(cache_dir, name)
    payload = {f"layer_{layer}": arr for layer, arr in acts.items()}
    payload["y"] = y
    np.savez_compressed(npz, **payload)
    meta.write_text(json.dumps({"n": int(len(y)), "layers": sorted(acts), "n_truncated": n_truncated}))


def _load_acts(cache_dir: Path, name: str) -> tuple[dict[int, np.ndarray], np.ndarray] | None:
    npz, _meta = _cache_paths(cache_dir, name)
    if not npz.exists():
        return None
    data = np.load(npz)
    y = data["y"]
    acts = {int(k.split("_", 1)[1]): data[k] for k in data.files if k.startswith("layer_")}
    return acts, y


def main() -> None:
    _load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--max-train", type=int, default=400)
    p.add_argument("--datasets", default=",".join(TRANSFER_SETS))
    p.add_argument("--max-per-dataset", type=int, default=None)
    p.add_argument("--probe", default=None)
    p.add_argument("--reuse-probe", action="store_true")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=2)
    args = p.parse_args()

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise SystemExit(
            "HF_TOKEN is not set. Put a token with access to google/gemma-3-27b-it "
            "in /workspace/.env or ravi/week1-dyl-vs-oa/dyl-probe-repro/.env "
            "(HF_TOKEN=hf_...)."
        )

    cfg = load_config(args.config)
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    unknown = [n for n in names if n not in TRANSFER_SETS]
    if unknown:
        raise SystemExit(f"unknown datasets: {unknown}; known: {list(TRANSFER_SETS)}")

    probe_path = Path(args.probe) if args.probe else cfg.results_dir / "oa_simple_deception_probe.json"
    cache_dir = cfg.results_dir / "oa_transfer_cache"
    out_path = cfg.results_dir / "oa_transfer_eval.json"

    subject = load_subject(
        cfg.model_name,
        dtype=cfg.dtype,
        device_map=cfg.device_map,
        attn_implementation=cfg.attn_implementation,
    )
    print(f"Model {subject.name}: {subject.n_layers} layers, d={subject.hidden_size}", flush=True)

    if args.reuse_probe and probe_path.exists():
        bundle = load_bundle(probe_path)
        print(f"Loaded OA probe from {probe_path} ({len(bundle.layers)} kept layers)", flush=True)
    else:
        examples = load_simple_deception(cfg, max_n=args.max_train + 80)
        train, _ = split_balanced(examples, eval_n=min(80, max(20, args.max_train // 10)))
        train = train[: args.max_train]
        print(f"OA train: {len(train)} simple-deception examples", flush=True)
        train_acts = extract_prefill_activations(
            subject,
            train,
            cfg.oa.followup_user,
            cfg.oa.followup_assistant,
            layers=None,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        y_train = np.array([ex.label for ex in train], dtype=np.int64)
        bundle = train_oa_bundle(
            train_acts,
            y_train,
            C=cfg.oa.C,
            ce_keep_below=cfg.oa.ce_keep_below,
            followup_user=cfg.oa.followup_user,
            followup_assistant=cfg.oa.followup_assistant,
            model_name=cfg.model_name,
            standardize=True,
            solver="liblinear",
        )
        save_bundle(bundle, probe_path)
        print(
            f"Kept {len(bundle.layers)} / {len(train_acts)} layers; wrote {probe_path}",
            flush=True,
        )

    payload = {
        "probe_path": str(probe_path),
        "trainer": "standardscaler_liblinear",
        "followup": {"user": bundle.followup_user, "assistant": bundle.followup_assistant},
        "kept_layers": bundle.layers,
        "datasets": {},
        "comparison": {},
    }

    for name in names:
        cached = _load_acts(cache_dir, name)
        if cached is None:
            convs = load_transfer_set(name, max_n=args.max_per_dataset)
            y = np.array([c.label for c in convs], dtype=np.int64)
            print(
                f"{name}: extracting {len(convs)} "
                f"({int((y == 1).sum())} lie / {int((y == 0).sum())} honest)",
                flush=True,
            )
            acts = extract_prefill_activations(
                subject,
                convs,
                bundle.followup_user,
                bundle.followup_assistant or "Yes",
                layers=sorted(bundle.probes),
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            _save_acts(cache_dir, name, acts, y, n_truncated=0)
        else:
            acts, y = cached
            print(f"{name}: loaded cached activations n={len(y)}", flush=True)

        metrics = _score_bundle(bundle, acts, y)
        payload["datasets"][name] = metrics
        known = KNOWN_DYL.get(name, {})
        l58 = metrics.get("layer_58", {}).get("auc")
        mean_auc = metrics.get("mean_kept", {}).get("auc")
        payload["comparison"][name] = {
            "oa_layer_58_auc": l58,
            "oa_mean_kept_auc": mean_auc,
            "dyl_probe_auc": known.get("dyl_probe"),
            "just_asking_auc": known.get("just_asking"),
            "oa_l58_gap_to_dyl": None if l58 is None or known.get("dyl_probe") is None else float(known["dyl_probe"] - l58),
            "n_honest": metrics.get("layer_58", metrics.get("mean_kept", {})).get("n_honest"),
            "n_deceptive": metrics.get("layer_58", metrics.get("mean_kept", {})).get("n_deceptive"),
            "dyl_note": known.get("note"),
        }
        write_json(out_path, payload)
        print(
            f"{name}: OA L58 AUC={l58}  mean-kept AUC={mean_auc}  "
            f"DYL probe={known.get('dyl_probe')}  asking={known.get('just_asking')}",
            flush=True,
        )

    print(json.dumps(payload["comparison"], indent=2), flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
