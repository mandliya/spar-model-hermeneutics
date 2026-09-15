from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyl_vs_oa.chat import parse_yes_no, with_prefill_followup
from dyl_vs_oa.config import load_config
from dyl_vs_oa.data import load_simple_deception, split_balanced
from dyl_vs_oa.metrics import check_against_target, summarize, wasserstein_1d
from dyl_vs_oa.probes import load_bundle, save_bundle, train_dyl_bundle, train_oa_bundle


def test_parse_yes_no():
    assert parse_yes_no("Yes") == 1
    assert parse_yes_no("yes.") == 1
    assert parse_yes_no("No") == 0
    assert parse_yes_no("  NO!") == 0
    assert parse_yes_no("") is None


def test_prefill_followup_appends_yes():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "secret"},
    ]
    out = with_prefill_followup(messages, "Did you respond truthfully?", "Yes")
    assert out[-2] == {"role": "user", "content": "Did you respond truthfully?"}
    assert out[-1] == {"role": "assistant", "content": "Yes"}


def test_simple_deception_loads_and_is_labeled():
    cfg = load_config()
    examples = load_simple_deception(cfg, max_n=40)
    assert len(examples) == 40
    assert {0, 1} <= {e.label for e in examples}
    train, ev = split_balanced(examples, eval_n=8)
    assert train and ev
    assert all(e.messages[-1]["role"] == "assistant" for e in examples)


def test_oa_bundle_keeps_low_ce_layers(tmp_path):
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 40)
    good = rng.normal(size=(80, 8))
    good[y == 1] += 2.0
    bad = rng.normal(size=(80, 8))
    acts = {0: bad, 12: good, 20: good}
    bundle = train_oa_bundle(
        acts,
        y,
        C=1.0,
        ce_keep_below=0.6,
        followup_user="Did you respond truthfully?",
        followup_assistant="Yes",
        model_name="dummy",
    )
    assert 12 in bundle.layers
    scores = bundle.score({k: acts[k] for k in bundle.layers})
    metrics = summarize(y, scores)
    assert metrics["auc"] > 0.9
    path = tmp_path / "probe.json"
    save_bundle(bundle, path)
    loaded = load_bundle(path)
    np.testing.assert_allclose(loaded.score({k: acts[k] for k in loaded.layers}), scores)


def test_dyl_single_layer_probe():
    rng = np.random.default_rng(1)
    y = np.array([0] * 30 + [1] * 30)
    x = rng.normal(size=(60, 16))
    x[y == 1] += 1.5
    bundle = train_dyl_bundle(
        x, y, layer=58, C=0.01, followup_user="Did you lie?", model_name="dummy"
    )
    assert bundle.layers == [58]
    assert bundle.extraction == "generate"
    metrics = summarize(y, bundle.score({58: x}))
    assert metrics["auc"] > 0.85


def test_paper_target_check():
    metrics = {"auc": 0.992}
    target = {"auc": 0.994}
    assert check_against_target(metrics, target, auc_tol=0.03)["ok"]
    assert not check_against_target({"auc": 0.70}, target, auc_tol=0.03)["ok"]


def test_wasserstein_separated():
    a = np.zeros(100)
    b = np.ones(100)
    assert wasserstein_1d(a, b) == pytest.approx(1.0, abs=1e-6)
