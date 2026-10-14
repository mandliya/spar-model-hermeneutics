"""Per-flavor matrix: folds, cutoffs, and the activation cache. No Gemma."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from noninvasive_probe.categories import FEW
from noninvasive_probe.flavor_matrix import (
    FOLDS,
    aisi_logits,
    cache_path,
    ensure_cached,
    family_of,
    fit_aisi_logistic,
    format_cell,
    matrix_for_position,
    out_of_fold_scores,
    render_position,
    require_flavor_cache,
    resolve_positions,
    result_signature,
    same_rows,
    stratified_folds,
    upsert_result,
    write_cache,
)
from noninvasive_probe.tianq_read import aisi_logits as logits_again


def _separable(n: int = 25, shift: float = 3.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    honest = rng.normal(size=(n, 4))
    lies = rng.normal(size=(n, 4)) + shift
    x = np.vstack([honest, lies])
    y = np.array([0] * n + [1] * n)
    return x, y


def test_out_of_fold_scores_match_disjoint_fits():
    x, y = _separable()
    folds = stratified_folds(y, folds=FOLDS, seed=0)
    assert len(folds) == 5
    manual = np.zeros(len(y))
    for train_idx, test_idx in folds:
        assert set(train_idx.tolist()).isdisjoint(test_idx.tolist())
        manual[test_idx] = aisi_logits(x[test_idx], *fit_aisi_logistic(x[train_idx], y[train_idx]))
    scores = out_of_fold_scores(x, y)
    assert np.allclose(scores, manual)
    assert roc_auc_score(y, scores) > 0.9


def test_few_rows_are_not_trained_and_stay_out_of_the_mean():
    alpha_x, alpha_y = _separable(n=20, seed=1)
    gamma_x, gamma_y = _separable(n=20, shift=-3.0, seed=2)
    small_x = np.random.default_rng(3).normal(size=(29, 4))
    small_y = np.array([1] * 9 + [0] * 20)
    rare_x = np.random.default_rng(4).normal(size=(27, 4))
    rare_y = np.array([1] * 20 + [0] * 7)
    assert 9 < FEW
    by_flavor = {
        "alpha": (alpha_x, alpha_y),
        "gamma": (gamma_x, gamma_y),
        "small": (small_x, small_y),
        "rare": (rare_x, rare_y),
    }
    matrix = matrix_for_position(
        by_flavor,
        position="mean_answer",
        train_flavors=["alpha", "gamma", "small"],
        test_flavors=["alpha", "gamma", "small", "rare"],
    )
    by_train = {row["train"]: row for row in matrix["rows"]}
    assert by_train["small"]["trained"] is False
    assert all(cell["kind"] == "not_trained" for cell in by_train["small"]["cells"])

    rare = next(cell for cell in by_train["alpha"]["cells"] if cell["test"] == "rare")
    assert rare["kind"] == "transfer" and rare["few"] is True and rare["n_honest"] == 7
    counted = [
        cell["auroc"]
        for row in matrix["rows"]
        for cell in row["cells"]
        if cell["kind"] == "transfer" and not cell["few"]
    ]
    assert rare["auroc"] not in counted
    assert matrix["mean_transfer"] == pytest.approx(float(np.mean(counted)))
    assert matrix["transfer_cells"] == len(counted)
    diagonal = next(cell for cell in by_train["alpha"]["cells"] if cell["test"] == "alpha")
    assert diagonal["kind"] == "in_category" and diagonal["few"] is False
    text = render_position(matrix, {"alpha": "Alpha", "gamma": "Gamma", "small": "Small", "rare": "Rare"})
    assert "**" in text
    assert "(few)" in text
    assert "not trained" in text
    assert "too few" in text


def test_fitter_does_not_load_instructed_pairs():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "src/noninvasive_probe/flavor_matrix.py",
        "src/noninvasive_probe/categories.py",
        "scripts/11_flavor_matrix.py",
    ):
        source = (root / relative).read_text()
        assert "instructed_pairs" not in source
        assert "simple_deception" not in source
    assert logits_again is aisi_logits


def test_cache_refuses_a_shorter_extract(tmp_path):
    uids = [str(i) for i in range(20)]
    y = np.array([i % 2 for i in range(20)])
    short = uids[:8]
    path = cache_path(tmp_path, "alibi", "closed", 4096)
    write_cache(path, short, y[:8], 4096, {"mean_answer": np.zeros((8, 3))})
    with pytest.raises(SystemExit, match="Refusing a smaller"):
        require_flavor_cache(path, uids, y, 4096, ["mean_answer"])
    saved = require_flavor_cache(path, short, y[:8], 4096, ["mean_answer"])
    assert saved["acts"]["mean_answer"].shape == (8, 3)
    assert same_rows(saved, short, y[:8], 4096)


def test_ensure_cached_skips_rows_already_stored(tmp_path):
    uids = ["a", "b"]
    y = np.array([0, 1])
    path = cache_path(tmp_path, "alibi", "tianq", 4096)
    calls = []

    def extract_missing(missing):
        calls.append(list(missing))
        return {position: np.ones((2, 3)) for position in missing}, 0

    ensure_cached(path, uids, y, 4096, ["reply_mean"], refresh=False, extract_missing=extract_missing)
    ensure_cached(path, uids, y, 4096, ["reply_mean"], refresh=False, extract_missing=extract_missing)
    assert calls == [["reply_mean"]]

    def extract_more(missing):
        calls.append(list(missing))
        return {position: np.full((2, 3), 2.0) for position in missing}, 0

    saved = ensure_cached(
        path, uids, y, 4096, ["reply_mean", "reply_last"], refresh=False, extract_missing=extract_more
    )
    assert calls[-1] == ["reply_last"]
    assert saved["acts"]["reply_mean"][0, 0] == pytest.approx(1.0)
    assert saved["acts"]["reply_last"][0, 0] == pytest.approx(2.0)


def test_upsert_keeps_finished_positions(tmp_path):
    path = tmp_path / "flavor_matrix.json"
    signature = result_signature(["alibi"], ["mean_answer", "reply_mean"], 4096, None)
    first = {"position": "mean_answer", "mean_in_category": 0.8, "mean_transfer": 0.5,
             "transfer_cells": 0, "transfer_cells_at_least_0_7": 0, "rows": [], "test": [], "train": []}
    upsert_result(path, signature, first, "mean answer table")
    second = dict(first, position="reply_mean")
    payload = upsert_result(path, signature, second, "reply mean table")
    assert set(payload["positions"]) == {"mean_answer", "reply_mean"}
    other = result_signature(["alibi"], ["mean_answer"], 8192, None)
    replaced = upsert_result(path, other, first, "replaced")
    assert set(replaced["positions"]) == {"mean_answer"}


def test_position_families_and_cell_text():
    assert family_of("mean_answer") == "closed"
    assert family_of("reply_mean") == "tianq"
    assert resolve_positions("reply_mean,mean_answer") == ["mean_answer", "reply_mean"]
    assert format_cell({"kind": "in_category", "auroc": 0.823, "n_lie": 43, "n_honest": 28, "few": False}) == "**0.82** (43/28)"
    assert format_cell({"kind": "transfer", "auroc": 0.51, "n_lie": 20, "n_honest": 7, "few": True}) == "0.51 (20/7) (few)"
