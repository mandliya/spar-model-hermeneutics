from __future__ import annotations

import numpy as np

from noninvasive_probe.data_acts import ACT_SOURCES, build_act_prefixes
from noninvasive_probe.heads import mean_unit_directions, orthogonal_component, train_ccs_direction
from noninvasive_probe.sets import EVAL_SETS, TRAIN_SOURCES
from noninvasive_probe.config import load_config


def test_act_prefixes_are_paired_policies_with_no_assistant_turn():
    cfg = load_config()
    rows = build_act_prefixes(cfg, n_pairs=4)
    assert len(rows) == 4 * 2 * 3
    sources = [r["source"] for r in rows]
    assert sources[:8] == ["onpolicy"] * 8
    for i in range(0, len(rows), 2):
        honest, deceptive = rows[i], rows[i + 1]
        assert honest["source"] == deceptive["source"]
        assert honest["label"] == 0 and deceptive["label"] == 1
        assert honest["user"] == deceptive["user"]
        assert honest["system"] != deceptive["system"]
        assert "assistant" not in honest
    assert set(ACT_SOURCES) <= TRAIN_SOURCES
    assert TRAIN_SOURCES.isdisjoint(EVAL_SETS)


def test_orthogonal_component_removes_the_truth_axis():
    direction = np.array([1.0, 1.0])
    out = orthogonal_component(direction, np.array([1.0, 0.0]))
    assert abs(out[0]) < 1e-8
    assert out[1] > 0


def test_mean_unit_directions_flips_opposites():
    out = mean_unit_directions([np.array([1.0, 0.0]), np.array([-2.0, 0.0]), np.array([0.0, 1.0])])
    assert out[0] > 0
    assert out[1] > 0


def test_ccs_points_toward_the_deceptive_side():
    rng = np.random.default_rng(0)
    honest = rng.normal(size=(40, 6))
    deceptive = honest + np.array([4.0, 0, 0, 0, 0, 0])
    vec, _mean, _std = train_ccs_direction(honest, deceptive, steps=250, seed=0)
    assert vec[0] > 0.5
