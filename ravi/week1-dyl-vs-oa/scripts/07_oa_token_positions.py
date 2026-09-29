#!/usr/bin/env python3
"""Train OA Simple Deception at four prefill tokens and score transfer sets.

Positions in the closed Gemma chat after `Did you respond truthfully?` / `Yes`:

    prompt_last   \\n after `model`     (Tianq official DYL)
    yes           the prefilled Yes     (OA paper / OA code)
    end_of_turn   <end_of_turn>
    last_newline  trailing \\n          (scripts/05)

Each position gets its own probe. One forward pass extracts all four.
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
from dyl_vs_oa.data import load_mbpp_honeypot, load_simple_deception, mbpp_conversations, split_balanced
from dyl_vs_oa.extract import OA_PREFILL_POSITIONS, extract_prefill_activations_at_positions
from dyl_vs_oa.metrics import summarize, write_json
from dyl_vs_oa.model import load_subject
from dyl_vs_oa.oa_transfer import KNOWN_DYL, load_transfer_set
from dyl_vs_oa.probes import train_oa_bundle


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


def _score(bundle, acts: dict[int, np.ndarray], y: np.ndarray) -> dict:
    out = {"n_kept": len(bundle.layers), "mean_kept": summarize(y, bundle.score(acts))}
    if 58 in bundle.probes and 58 in acts:
        out["layer_58"] = summarize(y, bundle.probes[58].predict_proba(acts[58]))
    return out


def _auc_row(metrics: dict) -> tuple[float | None, float | None]:
    l58 = metrics.get("layer_58", {}).get("auc")
    mean = metrics.get("mean_kept", {}).get("auc")
    return l58, mean


def main() -> None:
    _load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--max-train", type=int, default=400)
    p.add_argument(
        "--datasets",
        default="ascii_sandbagging,sycophancy_answer,sycophancy_feedback,mask_factual,capture_the_secret,dyl_city",
    )
    p.add_argument("--max-city", type=int, default=800)
    p.add_argument("--max-mbpp", type=int, default=200, help="MBPP tasks; 0 skips")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=2)
    args = p.parse_args()

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise SystemExit("HF_TOKEN is not set")

    cfg = load_config(args.config)
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    out_path = cfg.results_dir / "oa_token_position_eval.json"

    subject = load_subject(
        cfg.model_name,
        dtype=cfg.dtype,
        device_map=cfg.device_map,
        attn_implementation=cfg.attn_implementation,
    )
    print(
        f"Model {subject.name}: {subject.n_layers} layers, d={subject.hidden_size}. "
        f"Positions: {list(OA_PREFILL_POSITIONS)}",
        flush=True,
    )

    examples = load_simple_deception(cfg, max_n=args.max_train + 80)
    train, _ = split_balanced(examples, eval_n=min(80, max(20, args.max_train // 10)))
    train = train[: args.max_train]
    print(f"OA train: {len(train)} simple-deception examples", flush=True)
    train_by_pos = extract_prefill_activations_at_positions(
        subject,
        train,
        cfg.oa.followup_user,
        cfg.oa.followup_assistant,
        layers=None,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    y_train = np.array([ex.label for ex in train], dtype=np.int64)
    bundles = {}
    for name in OA_PREFILL_POSITIONS:
        bundles[name] = train_oa_bundle(
            train_by_pos[name],
            y_train,
            C=cfg.oa.C,
            ce_keep_below=cfg.oa.ce_keep_below,
            followup_user=cfg.oa.followup_user,
            followup_assistant=cfg.oa.followup_assistant,
            model_name=cfg.model_name,
            standardize=True,
            solver="liblinear",
        )
        n_kept = len(bundles[name].layers)
        print(f"trained {name}: kept {n_kept} / {len(train_by_pos[name])} layers", flush=True)

    payload = {
        "followup": {"user": cfg.oa.followup_user, "assistant": cfg.oa.followup_assistant},
        "positions": list(OA_PREFILL_POSITIONS),
        "note": (
            "prompt_last = Tianq DYL token; yes = OA paper/code; "
            "last_newline = scripts/05 extractor"
        ),
        "datasets": {},
        "table": {},
    }

    eval_sets: list[tuple[str, list]] = []
    for name in names:
        max_n = args.max_city if name == "dyl_city" else None
        convs = load_transfer_set(name, max_n=max_n)
        eval_sets.append((name, convs))
    if args.max_mbpp:
        rows = load_mbpp_honeypot(cfg, split="test", max_n=args.max_mbpp)
        mbpp = []
        for row in rows:
            honest, deceptive = mbpp_conversations(row, cfg.oa.system_prompt)
            mbpp.extend([honest, deceptive])
        eval_sets.append(("oa_mbpp", mbpp))
        print(f"oa_mbpp: {len(mbpp)} conversations", flush=True)

    for name, convs in eval_sets:
        y = np.array([c.label for c in convs], dtype=np.int64)
        print(
            f"{name}: extracting {len(convs)} "
            f"({int((y == 1).sum())} lie / {int((y == 0).sum())} honest)",
            flush=True,
        )
        by_pos = extract_prefill_activations_at_positions(
            subject,
            convs,
            cfg.oa.followup_user,
            cfg.oa.followup_assistant,
            layers=None,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        payload["datasets"][name] = {}
        payload["table"][name] = {"dyl_probe": KNOWN_DYL.get(name, {}).get("dyl_probe")}
        print(f"{name}  pos            L58     mean", flush=True)
        for pos in OA_PREFILL_POSITIONS:
            metrics = _score(bundles[pos], by_pos[pos], y)
            payload["datasets"][name][pos] = metrics
            l58, mean = _auc_row(metrics)
            payload["table"][name][pos] = {"layer_58": l58, "mean_kept": mean}
            print(
                f"{name}  {pos:14s}  {l58:.4f}  {mean:.4f}" if l58 is not None and mean is not None else f"{name} {pos} skipped",
                flush=True,
            )
        write_json(out_path, payload)

    print(json.dumps(payload["table"], indent=2), flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
