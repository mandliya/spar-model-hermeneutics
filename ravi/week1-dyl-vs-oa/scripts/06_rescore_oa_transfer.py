#!/usr/bin/env python3
"""Rescore OA transfer caches with clean original-turn labels.

City/varied npz files were extracted with the old inclusion rule (AISI
ambiguous-as-honest). Activations are still valid; only y was wrong.
This overwrites oa_transfer_eval.json with the clean numbers only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dyl-probe-repro"))

from dyl_vs_oa.chat import parse_yes_no
from dyl_vs_oa.metrics import summarize, write_json
from dyl_vs_oa.oa_transfer import (
    AISI_FOLLOWUP,
    KNOWN_DYL,
    _load_aisi_split,
    original_turn_label,
)
from dyl_vs_oa.probes import load_bundle

SPLITS = {
    "dyl_city": "dyl_train_city_countries",
    "dyl_varied": "dyl_validate_varied_deception",
}


def _old_included_label(is_lie: str, followup_answer: str) -> int | None:
    """Inclusion rule used when the existing npz files were written."""
    admitted = parse_yes_no(followup_answer)
    if is_lie == "lie":
        return 1
    if is_lie == "honest":
        if admitted in (0, 1):
            return 1 if admitted == 1 else 0
        return None
    if is_lie == "ambiguous" and admitted == 1:
        return 0
    return None


def aligned_clean_y(split: str, n_cache: int) -> np.ndarray:
    y = []
    for row in _load_aisi_split(split):
        raw = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
        messages = [{"role": m["role"], "content": m["content"]} for m in raw]
        if messages[-2] != {"role": "user", "content": AISI_FOLLOWUP}:
            raise ValueError("follow-up mismatch")
        old = _old_included_label(row["is_lie"], messages[-1]["content"])
        if old is None:
            continue
        y.append(
            original_turn_label(
                row["is_lie"], messages[-1]["content"], lie_reason=row.get("lie_reason")
            )
        )
    if len(y) != n_cache:
        raise SystemExit(f"{split}: aligned {len(y)} rows, cache has {n_cache}")
    return np.array(y, dtype=object)


def score(bundle, acts, y):
    mask = np.array([v is not None for v in y])
    y_i = np.array([int(v) for v, m in zip(y, mask) if m], dtype=np.int64)
    acts_m = {k: v[mask] for k, v in acts.items()}
    out = {
        "n_extracted": int(len(y)),
        "n_scored": int(mask.sum()),
        "n_dropped_ambiguous": int((~mask).sum()),
        "mean_kept": summarize(y_i, bundle.score(acts_m)),
    }
    if 58 in bundle.probes and 58 in acts_m:
        out["layer_58"] = summarize(y_i, bundle.probes[58].predict_proba(acts_m[58]))
    return out


def main() -> None:
    cfg_results = ROOT / "results"
    bundle = load_bundle(cfg_results / "oa_simple_deception_probe.json")
    payload = {
        "probe_path": str(cfg_results / "oa_simple_deception_probe.json"),
        "trainer": "standardscaler_liblinear",
        "label": "original-turn via AISI lie_reason; ambiguous dropped",
        "followup": {"user": bundle.followup_user, "assistant": bundle.followup_assistant},
        "datasets": {},
        "comparison": {},
    }
    existing = cfg_results / "oa_transfer_eval.json"
    if existing.exists():
        prev = json.loads(existing.read_text())
        for name, metrics in prev.get("datasets", {}).items():
            if name not in SPLITS:
                payload["datasets"][name] = metrics
                payload["comparison"][name] = prev.get("comparison", {}).get(name, {})

    cache_dir = cfg_results / "oa_transfer_cache"
    for name, split in SPLITS.items():
        npz = cache_dir / f"{name}.npz"
        if not npz.exists():
            continue
        data = np.load(npz)
        acts = {int(k.split("_", 1)[1]): data[k] for k in data.files if k.startswith("layer_")}
        y = aligned_clean_y(split, len(data["y"]))
        metrics = score(bundle, acts, y)
        payload["datasets"][name] = metrics
        l58 = metrics["layer_58"]["auc"]
        mean_auc = metrics["mean_kept"]["auc"]
        payload["comparison"][name] = {
            "oa_layer_58_auc": l58,
            "oa_mean_kept_auc": mean_auc,
            "n_honest": metrics["layer_58"]["n_honest"],
            "n_deceptive": metrics["layer_58"]["n_deceptive"],
            "n_dropped_ambiguous": metrics["n_dropped_ambiguous"],
            "note": KNOWN_DYL[name]["note"],
        }
        print(
            f"{name}: L58 {l58:.3f}  mean {mean_auc:.3f}  "
            f"{int(metrics['layer_58']['n_deceptive'])} lie / "
            f"{int(metrics['layer_58']['n_honest'])} honest  "
            f"(dropped {metrics['n_dropped_ambiguous']} ambiguous)",
            flush=True,
        )

    write_json(cfg_results / "oa_transfer_eval.json", payload)
    print(json.dumps(payload["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
