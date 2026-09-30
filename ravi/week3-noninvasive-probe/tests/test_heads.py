from __future__ import annotations

import numpy as np

from noninvasive_probe.config import load_config
from noninvasive_probe.data_intent import DECEPTIVE_INTENT_SYSTEM, HONEST_INTENT_SYSTEM, load_instructed_pairs
from noninvasive_probe.heads import train_lat_bundle, train_lat_direction, train_mean_diff_bundle


def test_instructed_pairs_alternate_same_statement():
    cfg = load_config()
    pairs = load_instructed_pairs(cfg, max_n=6)
    assert len(pairs) == 6
    assert [p.label for p in pairs] == [0, 1, 0, 1, 0, 1]
    assert pairs[0].messages[-1]["content"] == pairs[1].messages[-1]["content"]
    assert pairs[0].messages[0]["content"] == HONEST_INTENT_SYSTEM
    assert pairs[1].messages[0]["content"] == DECEPTIVE_INTENT_SYSTEM


def test_mean_diff_separates_shifted_gaussians():
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 40)
    x = rng.normal(size=(80, 8))
    x[y == 1] += 3.0
    bundle = train_mean_diff_bundle({3: x}, y, model_name="dummy")
    scores = bundle.logits({3: x})
    assert scores[y == 1].mean() > scores[y == 0].mean()
    assert bundle.extraction == "original_turn"
    assert bundle.method == "mean_diff"


def test_lat_flips_so_deceptive_is_positive():
    rng = np.random.default_rng(0)
    honest = rng.normal(size=(8, 4))
    deceptive = honest + 1.0
    vec = train_lat_direction(honest, deceptive)
    assert float(np.dot(deceptive.mean(axis=0) - honest.mean(axis=0), vec)) > 0
    y = np.array([0] * 8 + [1] * 8)
    x = np.vstack([honest, deceptive])
    bundle = train_lat_bundle({1: x}, y, model_name="dummy")
    scores = bundle.logits({1: x})
    assert scores[8:].mean() > scores[:8].mean()
