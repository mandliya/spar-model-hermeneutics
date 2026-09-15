#!/usr/bin/env python3
"""Load the official DYL probe, or train one from on-policy city/country rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dyl_vs_oa.chat import apply_chat, conversation_with_completion  # noqa: E402
from dyl_vs_oa.config import load_config
from dyl_vs_oa.data import LabeledConversation, dyl_training_prompts, load_city_countries
from dyl_vs_oa.extract import extract_generated_activations
from dyl_vs_oa.metrics import summarize, write_json
from dyl_vs_oa.model import load_subject
from dyl_vs_oa.official import load_official_dyl
from dyl_vs_oa.probes import save_bundle, train_dyl_bundle


def generate_rollouts(subject, specs, max_new_tokens: int, max_length: int):
    """On-policy DYL data: answer the city question, then keep the completion."""
    conversations = []
    for spec in specs:
        messages = []
        if spec["system"]:
            messages.append({"role": "system", "content": spec["system"]})
        messages.append({"role": "user", "content": spec["user"]})
        prompt = apply_chat(subject.tokenizer, messages, add_generation_prompt=True)
        enc = subject.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(subject.device) for k, v in enc.items()}
        gen = subject.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=subject.tokenizer.pad_token_id,
        )
        new_ids = gen[0, enc["input_ids"].shape[1] :].tolist()
        completion = subject.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        country = spec["country"].lower()
        said_true = country in completion.lower()
        if spec["intended"] == "honest":
            if not said_true:
                continue  # DYL: keep only rollouts where the model knows / says the true country
            label = 0
        else:
            if said_true:
                continue  # instructed to lie but didn't
            label = 1
        conv = LabeledConversation(
            messages=conversation_with_completion(spec["user"], completion, spec["system"]),
            label=label,
            meta={"city": spec["city"], "intended": spec["intended"], "completion": completion},
        )
        conversations.append(conv)
    return conversations


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--load-official", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--n-cities", type=int, default=None)
    p.add_argument("--max-rollouts", type=int, default=None)
    p.add_argument("--layer", type=int, default=None)
    args = p.parse_args()
    if not args.load_official and not args.train:
        raise SystemExit("Pass --load-official and/or --train")

    cfg = load_config(args.config)
    layer = args.layer or cfg.dyl.layer
    n_cities = args.n_cities or cfg.dyl.n_cities
    max_rollouts = args.max_rollouts or cfg.dyl.max_rollouts

    if args.load_official:
        bundle = load_official_dyl(cfg)
        path = cfg.results_dir / "dyl_official_probe.json"
        save_bundle(bundle, path)
        write_json(
            cfg.results_dir / "dyl_official_meta.json",
            {
                "repo": cfg.dyl.official_repo,
                "filename": cfg.dyl.official_filename,
                "layers": bundle.layers,
                "threshold": bundle.threshold,
                "coef_dim": int(bundle.probes[bundle.layers[0]].coef.size),
                "has_dataset_mean": bundle.probes[bundle.layers[0]].dataset_mean is not None,
                "has_dataset_std": bundle.probes[bundle.layers[0]].dataset_std is not None,
            },
        )
        print(f"Loaded official DYL probe layer={bundle.layers} -> {path}")

    if not args.train:
        return

    subject = load_subject(
        cfg.model_name,
        dtype=cfg.dtype,
        device_map=cfg.device_map,
        attn_implementation=cfg.attn_implementation,
    )
    cities = load_city_countries(cfg)[:n_cities]
    specs = dyl_training_prompts(cities)[: max_rollouts * 2]
    print(f"Generating up to {len(specs)} city/country rollouts")
    convs = generate_rollouts(subject, specs, max_new_tokens=32, max_length=cfg.max_sequence_length)
    print(f"Kept {len(convs)} rollouts after answer-quality filter")

    acts, answers, parsed = extract_generated_activations(
        subject,
        convs,
        cfg.dyl.followup_user,
        layer=layer,
        batch_size=1,
        max_new_tokens=cfg.gen_max_new_tokens,
        max_length=cfg.max_sequence_length,
    )
    y = np.array([c.label for c in convs], dtype=np.int64)
    yn_all = np.array([(-1 if p is None else p) for p in parsed], dtype=np.int64)
    # DYL: drop honest examples that falsely confess.
    keep = []
    for i, (label, yn) in enumerate(zip(y, yn_all)):
        if label == 0 and yn == 1:
            continue
        keep.append(i)
    keep = np.array(keep, dtype=np.int64)
    acts, y, yn_all = acts[keep], y[keep], yn_all[keep]
    answers = [answers[i] for i in keep]
    convs = [convs[i] for i in keep]
    print(f"After dropping honest false confessions: {len(y)} examples")
    print("label x self-report (1=yes I lied, 0=no, -1=unparsed):")
    for lab, lab_name in ((0, "honest"), (1, "deceptive")):
        for yn, yn_name in ((0, "no"), (1, "yes"), (-1, "unparsed")):
            n = int(((y == lab) & (yn_all == yn)).sum())
            if n:
                print(f"  {lab_name:10s} said {yn_name:8s}: {n}")

    # Held-out split. The previous 1.0 AUC was train-set interpolation (CE ~ 1e-7).
    rng = np.random.default_rng(0)
    hon_idx = np.where(y == 0)[0]
    dec_idx = np.where(y == 1)[0]
    rng.shuffle(hon_idx)
    rng.shuffle(dec_idx)
    n_val_each = min(40, len(hon_idx) // 5, len(dec_idx) // 5)
    val_idx = np.concatenate([hon_idx[:n_val_each], dec_idx[:n_val_each]])
    train_idx = np.concatenate([hon_idx[n_val_each:], dec_idx[n_val_each:]])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    bundle = train_dyl_bundle(
        acts[train_idx],
        y[train_idx],
        layer=layer,
        C=cfg.dyl.C,
        followup_user=cfg.dyl.followup_user,
        model_name=cfg.model_name,
    )
    save_bundle(bundle, cfg.results_dir / "dyl_trained_probe.json")
    train_metrics = summarize(y[train_idx], bundle.score({layer: acts[train_idx]}))
    val_metrics = summarize(y[val_idx], bundle.score({layer: acts[val_idx]}))
    self_report = yn_all[val_idx].astype(np.float64)
    self_report[self_report < 0] = 0.5
    self_report_metrics = summarize(y[val_idx], self_report)
    payload = {
        "train": train_metrics,
        "val": val_metrics,
        "self_report_val": self_report_metrics,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "layer": layer,
        "train_ce": bundle.probes[layer].train_ce,
        "answers_head": answers[:8],
    }
    write_json(cfg.results_dir / "dyl_train_metrics.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
