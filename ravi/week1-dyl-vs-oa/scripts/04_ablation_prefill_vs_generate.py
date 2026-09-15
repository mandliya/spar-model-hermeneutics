#!/usr/bin/env python3
"""Same Simple Deception pairs: OA prefill vs DYL generate. Isolates extraction, not data."""

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
from dyl_vs_oa.extract import extract_generated_activations, extract_prefill_activations
from dyl_vs_oa.metrics import summarize, write_json
from dyl_vs_oa.model import load_subject
from dyl_vs_oa.probes import save_bundle, train_dyl_bundle, train_oa_bundle


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--max-train", type=int, default=200)
    p.add_argument("--max-eval", type=int, default=100)
    p.add_argument("--dyl-layer", type=int, default=58)
    args = p.parse_args()

    cfg = load_config(args.config)
    examples = load_simple_deception(cfg, max_n=args.max_train + 40)
    train, _ = split_balanced(examples, eval_n=20)
    train = train[: args.max_train]
    y_train = np.array([e.label for e in train], dtype=np.int64)

    subject = load_subject(
        cfg.model_name,
        dtype=cfg.dtype,
        device_map=cfg.device_map,
        attn_implementation=cfg.attn_implementation,
    )

    prefill_acts = extract_prefill_activations(
        subject,
        train,
        cfg.oa.followup_user,
        cfg.oa.followup_assistant,
        layers=None,
        batch_size=cfg.activation_batch_size,
        max_length=cfg.max_sequence_length,
    )
    oa_bundle = train_oa_bundle(
        prefill_acts,
        y_train,
        C=cfg.oa.C,
        ce_keep_below=cfg.oa.ce_keep_below,
        followup_user=cfg.oa.followup_user,
        followup_assistant=cfg.oa.followup_assistant,
        model_name=cfg.model_name,
    )
    save_bundle(oa_bundle, cfg.results_dir / "ablation_oa_prefill.json")

    gen_acts, _, _ = extract_generated_activations(
        subject,
        train,
        cfg.dyl.followup_user,
        layer=args.dyl_layer,
        batch_size=1,
        max_new_tokens=cfg.gen_max_new_tokens,
        max_length=cfg.max_sequence_length,
    )
    dyl_bundle = train_dyl_bundle(
        gen_acts,
        y_train,
        layer=args.dyl_layer,
        C=cfg.dyl.C,
        followup_user=cfg.dyl.followup_user,
        model_name=cfg.model_name,
    )
    save_bundle(dyl_bundle, cfg.results_dir / "ablation_dyl_generate.json")

    rows = load_mbpp_honeypot(cfg, split="test", max_n=args.max_eval)
    convs = []
    for row in rows:
        hon, dec = mbpp_conversations(row, cfg.oa.system_prompt)
        convs.extend([hon, dec])
    y = np.array([c.label for c in convs], dtype=np.int64)

    oa_eval = extract_prefill_activations(
        subject,
        convs,
        oa_bundle.followup_user,
        oa_bundle.followup_assistant,
        layers=oa_bundle.layers,
        batch_size=cfg.activation_batch_size,
        max_length=cfg.max_sequence_length,
    )
    dyl_eval, _, _ = extract_generated_activations(
        subject,
        convs,
        dyl_bundle.followup_user,
        layer=args.dyl_layer,
        batch_size=1,
        max_new_tokens=cfg.gen_max_new_tokens,
        max_length=cfg.max_sequence_length,
    )
    payload = {
        "matched_training_data": "simple_deception",
        "oa_prefill_on_mbpp": summarize(y, oa_bundle.score(oa_eval)),
        "dyl_generate_on_mbpp": summarize(y, dyl_bundle.score({args.dyl_layer: dyl_eval})),
        "n_train": int(len(train)),
        "n_eval": int(len(convs)),
        "oa_layers": oa_bundle.layers,
        "dyl_layer": args.dyl_layer,
    }
    payload["winner_auc"] = (
        "oa_prefill"
        if payload["oa_prefill_on_mbpp"]["auc"] >= payload["dyl_generate_on_mbpp"]["auc"]
        else "dyl_generate"
    )
    write_json(cfg.results_dir / "ablation_prefill_vs_generate.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
