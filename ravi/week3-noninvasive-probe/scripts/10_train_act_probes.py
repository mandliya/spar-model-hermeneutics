#!/usr/bin/env python3
"""Act probes: on-policy lies, synthetic sycophancy, synthetic concealment.

The assistant text is generated. Labels are the policy. Gigi, OA, and AISI
are scored and never fit. See NOTES.md, the act-probe section.

Also fits CCS on instructed pairs and on the act pairs, and a direction that
is the mean of the three act directions with the truth direction removed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

WEEK3 = Path(__file__).resolve().parents[1]
WEEK1 = WEEK3.parent / "week1-dyl-vs-oa"
sys.path.insert(0, str(WEEK3 / "src"))
sys.path.insert(0, str(WEEK1 / "src"))
sys.path.insert(0, str(WEEK1 / "dyl-probe-repro"))

from dyl_vs_oa.chat import apply_chat
from dyl_vs_oa.data import load_simple_deception, split_balanced
from dyl_vs_oa.metrics import summarize, write_json
from dyl_vs_oa.model import load_subject
from dyl_vs_oa.probes import LayerProbe, ProbeBundle, save_bundle, train_oa_bundle
from noninvasive_probe.config import load_config
from noninvasive_probe.data_acts import (
    ACT_SOURCES,
    build_act_prefixes,
    load_rollouts,
    write_rollouts,
)
from noninvasive_probe.data_intent import load_instructed_pairs
from noninvasive_probe.extract import extract_original_turn_activations_at_positions
from noninvasive_probe.heads import (
    direction_cosine,
    mean_unit_directions,
    orthogonal_component,
    train_ccs_direction,
    train_mean_diff_bundle,
)
from noninvasive_probe.protocol import comparison_row
from noninvasive_probe.sets import assert_generic_train, load_eval_set

ACT_POSITIONS = ("last_content", "mean_excl_last_5", "mean_first_5", "last_user")
DEFAULT_DATASETS = (
    "ascii_sandbagging,sycophancy_answer,sycophancy_feedback,"
    "mask_factual,capture_the_secret,oa_mbpp,dyl_city,dyl_varied"
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


def _generate(subject, prefixes: list[dict], batch_size: int, max_new_tokens: int) -> list[dict]:
    tok = subject.tokenizer
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    eos = eot if isinstance(eot, int) and eot >= 0 else tok.eos_token_id
    rows = []
    for start in range(0, len(prefixes), batch_size):
        batch = prefixes[start : start + batch_size]
        texts = [
            apply_chat(
                tok,
                [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}],
                add_generation_prompt=True,
            )
            for row in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(subject.device) for k, v in enc.items()}
        out = subject.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=eos,
        )
        prompt_len = enc["attention_mask"].sum(dim=1)
        for i, row in enumerate(batch):
            new_ids = out[i, int(prompt_len[i].item()) :]
            text = tok.decode(new_ids, skip_special_tokens=True).strip()
            rows.append({**row, "completion": text})
        print(f"generated {min(start + batch_size, len(prefixes))}/{len(prefixes)}", flush=True)
    return rows


def _drop_bad_pairs(rows: list[dict]) -> tuple[list[dict], dict]:
    grouped: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        grouped[(row["source"], row["pair"])][row["label"]] = row
    kept = []
    dropped = {"empty": 0, "identical": 0, "incomplete": 0}
    for key in sorted(grouped):
        pair = grouped[key]
        if 0 not in pair or 1 not in pair:
            dropped["incomplete"] += 1
            continue
        a = pair[0]["completion"].strip()
        b = pair[1]["completion"].strip()
        if not a or not b:
            dropped["empty"] += 1
            continue
        if a == b:
            dropped["identical"] += 1
            continue
        kept.append(pair[0])
        kept.append(pair[1])
    return kept, dropped


def _pairs(acts: dict[int, np.ndarray], y: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y)
    if np.any(y[0::2] != 0) or np.any(y[1::2] != 1):
        raise SystemExit("CCS/LAT pairing expects honest, deceptive, honest, deceptive, ...")
    return {layer: (arr[0::2], arr[1::2]) for layer, arr in acts.items()}


def _bundle_from_coefs(
    coefs: dict[int, np.ndarray],
    *,
    mean: np.ndarray,
    std: np.ndarray,
    method: str,
    model_name: str,
) -> ProbeBundle:
    probes = {}
    for layer, coef in coefs.items():
        probes[layer] = LayerProbe(
            layer=layer,
            coef=np.asarray(coef, dtype=np.float64),
            intercept=0.0,
            C=0.0,
            train_ce=float("nan"),
            dataset_mean=mean,
            dataset_std=std,
        )
    return ProbeBundle(
        method=method,
        layers=sorted(probes),
        probes=probes,
        followup_user="",
        followup_assistant=None,
        extraction="original_turn",
        model_name=model_name,
    )


def _raw_direction(probe: LayerProbe) -> np.ndarray:
    std = np.where(probe.dataset_std < 1e-8, 1.0, probe.dataset_std)
    return probe.coef / std


def _ccs_bundle(acts, y, model_name: str) -> ProbeBundle:
    probes = {}
    for layer, (honest, deceptive) in _pairs(acts, y).items():
        vec, mean, std = train_ccs_direction(honest, deceptive)
        probes[layer] = LayerProbe(
            layer=layer,
            coef=vec,
            intercept=0.0,
            C=0.0,
            train_ce=float("nan"),
            dataset_mean=mean,
            dataset_std=std,
        )
    return ProbeBundle(
        method="ccs",
        layers=sorted(probes),
        probes=probes,
        followup_user="",
        followup_assistant=None,
        extraction="original_turn",
        model_name=model_name,
    )


def _score(bundle, acts, y) -> dict:
    out = {"n_kept": len(bundle.layers), "mean_kept": summarize(y, bundle.score(acts))}
    if 58 in bundle.probes and 58 in acts:
        out["layer_58"] = summarize(y, bundle.probes[58].predict_proba(acts[58]))
    return out


def _cache_path(cache_dir: Path, dataset: str, position: str) -> Path:
    return cache_dir / f"{dataset}__{position}.npz"


def _load_cache(cache_dir: Path, dataset: str, position: str):
    path = _cache_path(cache_dir, dataset, position)
    if not path.exists():
        return None
    data = np.load(path)
    y = data["y"]
    acts = {int(k.split("_", 1)[1]): data[k] for k in data.files if k.startswith("layer_")}
    return acts, y


def _save_cache(cache_dir: Path, dataset: str, position: str, acts, y, uids) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {f"layer_{layer}": arr for layer, arr in acts.items()}
    payload["y"] = y
    np.savez_compressed(_cache_path(cache_dir, dataset, position), **payload)
    (cache_dir / f"{dataset}__{position}.meta.json").write_text(
        json.dumps({"n": int(len(y)), "uids": uids, "layers": sorted(acts)})
    )


def _subset(acts, y, mask):
    return {layer: arr[mask] for layer, arr in acts.items()}, y[mask]


def main() -> None:
    _load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs", type=int, default=80)
    p.add_argument("--positions", default=",".join(ACT_POSITIONS))
    p.add_argument("--datasets", default=DEFAULT_DATASETS)
    p.add_argument("--max-per-dataset", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--extract-batch-size", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--skip-generate", action="store_true")
    args = p.parse_args()

    cfg = load_config(None)
    positions = tuple(x.strip() for x in args.positions.split(",") if x.strip())
    names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    rollout_path = cfg.results_dir / "act_rollouts.jsonl"
    probe_dir = cfg.results_dir / "act_probes"
    cache_dir = cfg.results_dir / "noninvasive_cache"
    probe_dir.mkdir(parents=True, exist_ok=True)

    prefixes = build_act_prefixes(cfg, args.n_pairs)
    subject = None
    if not args.skip_generate or not rollout_path.exists():
        if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            raise SystemExit("HF_TOKEN is not set")
        subject = load_subject(cfg.model_name, dtype=cfg.dtype, device_map=cfg.device_map, attn_implementation=cfg.attn_implementation)
        print(f"Model {subject.name}: {subject.n_layers} layers", flush=True)
        generated = _generate(subject, prefixes, args.batch_size, args.max_new_tokens)
        kept, dropped = _drop_bad_pairs(generated)
        write_rollouts(rollout_path, kept)
        print(f"rollouts kept={len(kept)} dropped={dropped} -> {rollout_path}", flush=True)
    else:
        dropped = None

    rollouts = load_rollouts(rollout_path)
    assert_generic_train(rollouts)
    y_acts = np.array([ex.label for ex in rollouts], dtype=np.int64)
    print(f"act rows {len(rollouts)} sources {sorted({ex.meta['source'] for ex in rollouts})}", flush=True)

    truth = load_simple_deception(cfg, max_n=480)
    truth, _ = split_balanced(truth, eval_n=80)
    truth = truth[:400]
    y_truth = np.array([ex.label for ex in truth], dtype=np.int64)
    intent = load_instructed_pairs(cfg, max_n=400)
    y_intent = np.array([ex.label for ex in intent], dtype=np.int64)

    if subject is None:
        subject = load_subject(cfg.model_name, dtype=cfg.dtype, device_map=cfg.device_map, attn_implementation=cfg.attn_implementation)
        print(f"Model {subject.name}: {subject.n_layers} layers", flush=True)

    print("extracting act rollouts", flush=True)
    act_acts = extract_original_turn_activations_at_positions(
        subject, rollouts, batch_size=args.extract_batch_size, max_length=args.max_length, positions=positions
    )
    print("extracting truth and instructed pairs", flush=True)
    truth_acts = extract_original_turn_activations_at_positions(
        subject, truth, batch_size=args.extract_batch_size, max_length=args.max_length, positions=positions
    )
    intent_acts = extract_original_turn_activations_at_positions(
        subject, intent, batch_size=args.extract_batch_size, max_length=args.max_length, positions=positions
    )

    bundles: dict[str, ProbeBundle] = {}
    cosines: dict[str, dict] = {}
    for position in positions:
        y = y_acts
        acts = act_acts[position]
        logistic = train_oa_bundle(
            acts, y, C=cfg.oa.C, ce_keep_below=cfg.oa.ce_keep_below,
            followup_user="", followup_assistant="", model_name=cfg.model_name,
            standardize=True, solver="liblinear",
        )
        logistic = replace(logistic, method="acts_logistic", extraction="original_turn", followup_assistant=None)
        mean_diff = train_mean_diff_bundle(acts, y, method="acts_mean_diff", model_name=cfg.model_name)
        ccs = _ccs_bundle(acts, y, cfg.model_name)
        ccs = replace(ccs, method="acts_ccs")
        intent_ccs = _ccs_bundle(intent_acts[position], y_intent, cfg.model_name)
        intent_ccs = replace(intent_ccs, method="intent_ccs")
        truth_md = train_mean_diff_bundle(truth_acts[position], y_truth, method="truth_mean_diff", model_name=cfg.model_name)
        source_md = {}
        for source in ACT_SOURCES:
            mask = np.array([ex.meta["source"] == source for ex in rollouts])
            sub_acts, sub_y = _subset(acts, y, mask)
            source_md[source] = train_mean_diff_bundle(
                sub_acts, sub_y, method=f"{source}_mean_diff", model_name=cfg.model_name
            )
        shared_coefs = {}
        orth_coefs = {}
        layer_cos = {}
        for layer in mean_diff.layers:
            raws = [_raw_direction(source_md[s].probes[layer]) for s in ACT_SOURCES]
            shared_raw = mean_unit_directions(raws)
            truth_raw = _raw_direction(truth_md.probes[layer])
            orth_raw = orthogonal_component(shared_raw, truth_raw)
            std = mean_diff.probes[layer].dataset_std
            shared_coefs[layer] = shared_raw * std
            orth_coefs[layer] = orth_raw * std
            if layer == 58:
                layer_cos = {
                    "onpolicy_syc": direction_cosine(raws[0], raws[1]),
                    "onpolicy_conceal": direction_cosine(raws[0], raws[2]),
                    "syc_conceal": direction_cosine(raws[1], raws[2]),
                    "shared_vs_truth": direction_cosine(shared_raw, truth_raw),
                    "acts_logistic_vs_truth": direction_cosine(
                        _raw_direction(logistic.probes[layer]), truth_raw
                    ),
                }
        ref = mean_diff.probes[mean_diff.layers[0]]
        shared = _bundle_from_coefs(
            shared_coefs, mean=ref.dataset_mean, std=ref.dataset_std,
            method="acts_shared", model_name=cfg.model_name,
        )
        # Per-layer mean/std differ. Rebuild with each layer's own scale.
        for bundle, coefs in ((shared, shared_coefs),):
            for layer, coef in coefs.items():
                base = mean_diff.probes[layer]
                bundle.probes[layer] = LayerProbe(
                    layer=layer, coef=coef, intercept=0.0, C=0.0, train_ce=float("nan"),
                    dataset_mean=base.dataset_mean, dataset_std=base.dataset_std,
                )
        orth = _bundle_from_coefs(
            orth_coefs, mean=ref.dataset_mean, std=ref.dataset_std,
            method="acts_orth_truth", model_name=cfg.model_name,
        )
        for layer, coef in orth_coefs.items():
            base = mean_diff.probes[layer]
            orth.probes[layer] = LayerProbe(
                layer=layer, coef=coef, intercept=0.0, C=0.0, train_ce=float("nan"),
                dataset_mean=base.dataset_mean, dataset_std=base.dataset_std,
            )
        named = {
            f"acts__{position}__logistic": logistic,
            f"acts__{position}__mean_diff": mean_diff,
            f"acts__{position}__ccs": ccs,
            f"intent__{position}__ccs": intent_ccs,
            f"acts__{position}__shared": shared,
            f"acts__{position}__orth_truth": orth,
        }
        for source, bundle in source_md.items():
            named[f"{source}__{position}__mean_diff"] = bundle
        for key, bundle in named.items():
            save_bundle(bundle, probe_dir / f"{key}.json")
            bundles[key] = bundle
        cosines[position] = layer_cos
        print(f"{position} L58 cosines {layer_cos}", flush=True)

    truth_acts = intent_acts = None
    report = {
        "n_pairs_requested": args.n_pairs,
        "n_act_rows": len(rollouts),
        "dropped_pairs": dropped,
        "positions": list(positions),
        "cosines_layer_58_raw": cosines,
        "examples": [],
        "datasets": {},
    }
    for source in ACT_SOURCES:
        row = next(r for r in rollouts if r.meta["source"] == source and r.label == 1)
        report["examples"].append({
            "source": source,
            "deceptive_completion": row.messages[-1]["content"][:400],
        })

    for name in names:
        convs = load_eval_set(name, cfg, max_n=args.max_per_dataset)
        y = np.array([c.label for c in convs], dtype=np.int64)
        uids = [str(c.meta.get("uid", i)) for i, c in enumerate(convs)]
        missing = [pos for pos in positions if _load_cache(cache_dir, name, pos) is None]
        extracted = {}
        if missing:
            print(f"{name}: extracting {len(convs)} at {missing}", flush=True)
            extracted = extract_original_turn_activations_at_positions(
                subject, convs, batch_size=args.extract_batch_size, max_length=args.max_length,
                positions=tuple(missing),
            )
            for pos, acts in extracted.items():
                _save_cache(cache_dir, name, pos, acts, y, uids)
        cell = {"n": int(len(y)), "positions": {}}
        known = comparison_row(name)
        cell["comparison"] = {
            "just_asking": known.get("just_asking"),
            "dyl_probe": known.get("dyl_probe"),
            "oa_layer_58": known.get("oa_layer_58"),
            "oa_mean": known.get("oa_mean"),
            "n_lie": int((y == 1).sum()),
            "n_honest": int((y == 0).sum()),
        }
        for position in positions:
            cached = _load_cache(cache_dir, name, position)
            acts, y_cached = cached
            pos_out = {}
            for key, bundle in bundles.items():
                if f"__{position}__" not in key:
                    continue
                pos_out[key] = _score(bundle, acts, y_cached)
            cell["positions"][position] = pos_out
        report["datasets"][name] = cell
        write_json(cfg.results_dir / "act_probe_eval.json", report)
        print(f"{name}: scored", flush=True)

    print(f"wrote {cfg.results_dir / 'act_probe_eval.json'}", flush=True)


if __name__ == "__main__":
    main()
