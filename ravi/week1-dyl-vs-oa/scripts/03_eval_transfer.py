#!/usr/bin/env python3
"""Compare OA probe, DYL probe, and black-box self-report on OA MBPP-Honeypot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dyl_vs_oa.chat import apply_chat, parse_yes_no, with_generate_followup
from dyl_vs_oa.config import load_config
from dyl_vs_oa.data import load_mbpp_honeypot, mbpp_conversations
from dyl_vs_oa.extract import extract_generated_activations, extract_prefill_activations
from dyl_vs_oa.metrics import summarize, write_json
from dyl_vs_oa.model import load_subject
from dyl_vs_oa.official import official_callable, score_official_logits
from dyl_vs_oa.probes import load_bundle


def self_report_scores(subject, conversations, followup_user: str, max_new_tokens: int, max_length: int):
    """Black-box baseline: ask the model if it lied. Score = P(parsed yes)."""
    scores = []
    answers = []
    for conv in conversations:
        messages = with_generate_followup(conv.messages, followup_user)
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
        text = subject.tokenizer.decode(new_ids, skip_special_tokens=True)
        answers.append(text)
        parsed = parse_yes_no(text)
        scores.append(1.0 if parsed == 1 else 0.0 if parsed == 0 else 0.5)
    return np.array(scores), answers


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--max-eval", type=int, default=None)
    p.add_argument("--oa-probe", default=None)
    p.add_argument("--dyl-probe", default=None)
    p.add_argument("--skip-self-report", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    max_eval = args.max_eval or cfg.oa.max_eval
    oa_path = Path(args.oa_probe) if args.oa_probe else cfg.results_dir / "oa_simple_deception_probe.json"
    dyl_path = Path(args.dyl_probe) if args.dyl_probe else None
    if dyl_path is None:
        official = cfg.results_dir / "dyl_official_probe.json"
        trained = cfg.results_dir / "dyl_trained_probe.json"
        dyl_path = official if official.exists() else trained

    oa_bundle = load_bundle(oa_path) if oa_path.exists() else None
    dyl_bundle = load_bundle(dyl_path) if dyl_path and dyl_path.exists() else None
    if oa_bundle is None and dyl_bundle is None:
        raise SystemExit("No probes found. Run 01_train_oa_probe.py and 02_train_dyl_probe.py first.")

    subject = load_subject(
        cfg.model_name,
        dtype=cfg.dtype,
        device_map=cfg.device_map,
        attn_implementation=cfg.attn_implementation,
    )
    rows = load_mbpp_honeypot(cfg, split="test", max_n=max_eval)
    convs = []
    for row in rows:
        hon, dec = mbpp_conversations(row, cfg.oa.system_prompt)
        convs.extend([hon, dec])
    y = np.array([c.label for c in convs], dtype=np.int64)
    print(f"Eval conversations: {len(convs)} ({int((y == 0).sum())} honest, {int((y == 1).sum())} deceptive)")

    results = {"n": int(len(convs)), "max_eval_tasks": max_eval}

    if oa_bundle is not None:
        acts = extract_prefill_activations(
            subject,
            convs,
            oa_bundle.followup_user,
            oa_bundle.followup_assistant or "Yes",
            layers=oa_bundle.layers,
            batch_size=cfg.activation_batch_size,
            max_length=cfg.max_sequence_length,
        )
        scores = oa_bundle.score(acts)
        results["oa_prefill"] = summarize(y, scores)
        print("OA prefill", results["oa_prefill"])

    if dyl_bundle is not None:
        layer = dyl_bundle.layers[0]
        acts, answers, parsed = extract_generated_activations(
            subject,
            convs,
            dyl_bundle.followup_user,
            layer=layer,
            batch_size=1,
            max_new_tokens=cfg.gen_max_new_tokens,
            max_length=cfg.max_sequence_length,
        )
        if dyl_bundle.method == "dyl-official":
            live = official_callable(cfg)
            scored = score_official_logits(live, acts)
            bundle_logits = dyl_bundle.logits({layer: acts})
            if not np.allclose(scored, bundle_logits, rtol=1e-4, atol=1e-3):
                max_abs = float(np.max(np.abs(scored - bundle_logits)))
                print(f"warning: unpacked official probe logits differ from live (max |Δ|={max_abs:.4g})")
            threshold = float(
                dyl_bundle.threshold
                if dyl_bundle.threshold is not None
                else np.asarray(live.threshold).reshape(-1)[0]
            )
            results["dyl_generate"] = summarize(y, scored, decision_threshold=threshold)
            results["dyl_generate"]["score_space"] = "logit"
        else:
            scored = dyl_bundle.score({layer: acts})
            results["dyl_generate"] = summarize(y, scored)
            results["dyl_generate"]["score_space"] = "probability"
        self_report = np.array([1.0 if p == 1 else 0.0 if p == 0 else 0.5 for p in parsed])
        results["dyl_self_report_from_generation"] = summarize(y, self_report)
        print("DYL generate", results["dyl_generate"])
        print(
            "DYL score range",
            {
                "min": float(np.min(scored)),
                "max": float(np.max(scored)),
                "mean": float(np.mean(scored)),
            },
        )

    if not args.skip_self_report:
        if dyl_bundle is not None:
            # Same follow-up as the generate pass above; don't pay for a second rollout.
            results["bb_did_you_lie"] = results["dyl_self_report_from_generation"]
            results["bb_did_you_lie_answers_head"] = answers[:6]
        else:
            followup = cfg.dyl.followup_user
            scores, answers = self_report_scores(
                subject, convs, followup, cfg.gen_max_new_tokens, cfg.max_sequence_length
            )
            results["bb_did_you_lie"] = summarize(y, scores)
            results["bb_did_you_lie_answers_head"] = answers[:6]
        print("BB did-you-lie", results["bb_did_you_lie"])

    if "oa_prefill" in results and "dyl_generate" in results:
        results["winner_auc"] = (
            "oa" if results["oa_prefill"]["auc"] >= results["dyl_generate"]["auc"] else "dyl"
        )
        results["beats_bb"] = {
            "oa": results["oa_prefill"]["auc"] > results.get("bb_did_you_lie", {}).get("auc", 0),
            "dyl": results["dyl_generate"]["auc"] > results.get("bb_did_you_lie", {}).get("auc", 0),
        }

    write_json(cfg.results_dir / "transfer_oa_rh.json", results)
    print(json.dumps({k: v for k, v in results.items() if k != "bb_did_you_lie_answers_head"}, indent=2))


if __name__ == "__main__":
    main()
