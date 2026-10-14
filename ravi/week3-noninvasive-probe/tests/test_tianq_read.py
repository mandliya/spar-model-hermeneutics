"""Tianq's reply positions and his logistic, without Gemma."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from noninvasive_probe.tianq_read import aisi_logits, fit_aisi_logistic, reply_token_ids


class _Tok:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert add_generation_prompt and tokenize
        assert messages[-1]["role"] != "assistant"
        return [10, 11, 12]

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [20, 21, 22, 23]}


def test_reply_indices_match_tianq_prefill():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "the reply"},
    ]
    ids, positions, span = reply_token_ids(_Tok(), messages)
    assert ids == [10, 11, 12, 20, 21, 22, 23]
    assert positions["prompt_last"] == 2
    assert positions["reply_last"] == 6
    assert span == (3, 7)


def test_aisi_logistic_separates_shifted_clusters():
    rng = np.random.default_rng(0)
    honest = rng.normal(size=(40, 8))
    lies = rng.normal(size=(40, 8)) + 2.5
    x = np.vstack([honest, lies])
    y = np.array([0] * 40 + [1] * 40)
    mean, std, weight, bias = fit_aisi_logistic(x, y)
    scores = aisi_logits(x, mean, std, weight, bias)
    assert roc_auc_score(y, scores) > 0.95
    assert scores[y == 1].mean() > scores[y == 0].mean()
