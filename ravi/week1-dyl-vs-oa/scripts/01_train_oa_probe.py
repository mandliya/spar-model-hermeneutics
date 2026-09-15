#!/usr/bin/env python3
"""Train an OA-style prefill probe on Simple Deception and eval on synthetic MBPP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dyl_vs_oa.config import load_config
from dyl_vs_oa.data import load_mbpp_honeypot, load_simple_deception, mbpp_conversations, split_balanced
from dyl_vs_oa.extract import extract_prefill_activations
from dyl_vs_oa.metrics import check_against_target, summarize, write_json
from dyl_vs_oa.model import load_subject
from dyl_vs_oa.probes import save_bundle, train_oa_bundle


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-eval", type=int, default=None)
    p.add_argument("--layers", default="all", help="'all' or comma-separated indices")
    args = p.parse_args()

    cfg = load_config(args.config)
    max_train = args.max_train or cfg.oa.max_train
    max_eval = args.max_eval or cfg.oa.max_eval

    examples = load_simple_deception(cfg, max_n=max_train + 80)
    train, _heldout = split_balanced(examples, eval_n=min(80, max(20, max_train // 10)))
    train = train[:max_train]
    print(f"OA train: {len(train)} simple-deception examples")

    subject = load_subject(
        cfg.model_name,
        dtype=cfg.dtype,
        device_map=cfg.device_map,
        attn_implementation=cfg.attn_implementation,
    )
    layers = None if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    print(f"Model {subject.name}: {subject.n_layers} layers, d={subject.hidden_size}")

    train_acts = extract_prefill_activations(
        subject,
        train,
        cfg.oa.followup_user,
        cfg.oa.followup_assistant,
        layers=layers,
        batch_size=cfg.activation_batch_size,
        max_length=cfg.max_sequence_length,
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
    )
    print(f"Kept {len(bundle.layers)} / {len(train_acts)} layers with CE < {cfg.oa.ce_keep_below}")
    print("layer CE:", {k: round(v.train_ce, 3) for k, v in sorted(bundle.probes.items())})

    out_probe = cfg.results_dir / "oa_simple_deception_probe.json"
    save_bundle(bundle, out_probe)
    print(f"Wrote {out_probe}")

    mbpp_rows = load_mbpp_honeypot(cfg, split="test", max_n=max_eval)
    eval_convs = []
    for row in mbpp_rows:
        hon, dec = mbpp_conversations(row, cfg.oa.system_prompt)
        eval_convs.extend([hon, dec])
    y_eval = np.array([ex.label for ex in eval_convs], dtype=np.int64)
    eval_acts = extract_prefill_activations(
        subject,
        eval_convs,
        cfg.oa.followup_user,
        cfg.oa.followup_assistant,
        layers=bundle.layers,
        batch_size=cfg.activation_batch_size,
        max_length=cfg.max_sequence_length,
    )
    scores = bundle.score(eval_acts)
    metrics = summarize(y_eval, scores)
    targets = json.loads((cfg.results_dir / "paper_targets.json").read_text())
    check = check_against_target(metrics, targets["oa_simple_deception_on_mbpp_synth"])
    payload = {
        "metrics": metrics,
        "paper_check": check,
        "n_train": len(train),
        "kept_layers": bundle.layers,
    }
    write_json(cfg.results_dir / "oa_simple_deception_mbpp_eval.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
