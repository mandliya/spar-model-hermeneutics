"""Results already on the Hub, read across targets and readers, and pushing this experiment's own.

A target's results live under its namespace (Gemma at the repo root); a reader's under
"<target>/read_by_<reader>" (dyl_repro/core/store.py). Result files are named after the probe layer.
"""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from dyl_repro.core.model import DEFAULT, MODELS

HUB = "yuqi-questions"
BOOTSTRAP = 2000
TARGETS = ("gemma-3-27b-it", "qwen3-32b", "qwen3-next-80b", "qwen3-235b")
LABELS = {"gemma-3-27b-it": "Gemma-3-27B", "qwen3-32b": "Qwen3-32B", "qwen3-next-80b": "Qwen3-Next-80B",
          "qwen3-235b": "Qwen3-235B", "gemma-3-12b-it": "Gemma-3-12B", "qwen3.5-27b": "Qwen3.5-27B"}


def namespace(target: str, reader: str | None = None) -> str:
    """Where a target's (or a reader of it's) results are stored, with a trailing slash."""
    base = "" if target == DEFAULT else f"{target}/"
    return base if reader is None or reader == target else f"{target}/read_by_{reader}/"


def layer_file(prefix: str, model: str) -> str:
    layer = MODELS[model].probe_layer
    return f"{prefix}_l_{layer}_lm_0_01_ar_lr_hs{layer + 1}.json"


@cache
def fetch(path: str) -> dict:
    return json.loads(Path(hf_hub_download(os.environ["HF_DATASET_REPO"], path, repo_type="dataset",
                                           token=os.environ["HF_TOKEN"])).read_text(encoding="utf-8"))


def diversity(target: str, reader: str | None = None) -> dict:
    """The k-curve result (cross_category_transfer/compare_diversity.py) of a target, or of a reader of it."""
    return fetch(f"{namespace(target, reader)}gigi/results/{layer_file('diversity', reader or target)}")


def faithfulness_gaps(target: str) -> dict:
    """Same-kind target-minus-reader gaps (hermeneutics/compare_faithfulness.py)."""
    return fetch(f"{namespace(target)}gigi/results/faithfulness_gaps.json")


def k_curve(result: dict, variant: str, line: str = "size_matched") -> list[dict]:
    """Held-out AUROC per k: the mean over the training subsets, and its 95% interval from resampling
    the subsets (seeded)."""
    rng = np.random.default_rng(0)
    out = []
    for k in sorted({r["k"] for r in result["records"]}):
        values = np.array([r["held_out_mean"] for r in result["records"]
                           if r["variant"] == variant and r["line"] == line and r["k"] == k
                           and r.get("held_out_mean") is not None])
        if not len(values):
            continue
        boots = rng.choice(values, (BOOTSTRAP, len(values))).mean(axis=1)
        out.append({"k": k, "mean": float(values.mean()), "lo": float(np.percentile(boots, 2.5)),
                    "hi": float(np.percentile(boots, 97.5)), "subsets": len(values)})
    return out


def subset_means(result: dict, variant: str, k: int, line: str = "size_matched") -> dict[tuple, float]:
    """Held-out mean per training subset at one k, keyed by the subset, for pairing two models."""
    return {tuple(r["subset"]): r["held_out_mean"] for r in result["records"]
            if r["variant"] == variant and r["line"] == line and r["k"] == k and r.get("held_out_mean") is not None}


def push(path: Path, folder: str) -> None:
    """One output to the Hub under yuqi-questions/<folder>/."""
    HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj=str(path), path_in_repo=f"{HUB}/{folder}/{path.name}",
                                                    repo_id=os.environ["HF_DATASET_REPO"], repo_type="dataset")
