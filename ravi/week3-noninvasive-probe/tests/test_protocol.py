from __future__ import annotations

from dyl_vs_oa.data import LabeledConversation
from noninvasive_probe.protocol import comparison_row, followup_text, htc_mask
from noninvasive_probe.sets import EVAL_SETS, TRAIN_SOURCES, admitted_from_reason, assert_generic_train


def test_htc_mask_keeps_denials_only():
    y = [1, 1, 0, 0, 1]
    admitted = [False, True, False, None, False]
    keep, counts = htc_mask(y, admitted)
    assert keep == [0, 2, 4]
    assert counts == {"n_lied_denied": 2, "n_truthful_denied": 1, "n_htc": 3}


def test_aisi_admitted_from_lie_reason():
    assert admitted_from_reason("Model denied deception (hiding it).") is False
    assert admitted_from_reason("Model honestly admitted deception.") is True
    assert admitted_from_reason("Model correctly denied deception.") is False
    assert admitted_from_reason("False confession: said yes after an honest turn.") is True
    assert admitted_from_reason("something else") is None


def test_named_followup_for_ascii_is_sandbag_not_generic_lie():
    assert "sandbag" in followup_text("ascii_sandbagging", "named").lower()
    assert "cheat" in followup_text("oa_mbpp", "named").lower()


def test_comparison_row_includes_week1_oa_baselines():
    ascii_row = comparison_row("ascii_sandbagging")
    assert ascii_row["dyl_probe"] == 0.99
    assert ascii_row["oa_layer_58"] == 0.996
    city = comparison_row("dyl_city")
    assert city["oa_mean"] == 0.965
    assert city["dyl_probe"] is None


def test_train_sources_are_disjoint_from_eval_sets():
    assert TRAIN_SOURCES.isdisjoint(EVAL_SETS)
    from dyl_vs_oa.data import load_simple_deception
    from noninvasive_probe.config import load_config
    from noninvasive_probe.data_intent import load_instructed_pairs

    cfg = load_config()
    assert_generic_train(load_simple_deception(cfg, max_n=8))
    assert_generic_train(load_instructed_pairs(cfg, max_n=4))
    leak = LabeledConversation(
        messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        label=1,
        meta={"source": "ascii_sandbagging"},
    )
    try:
        assert_generic_train([leak])
    except ValueError as err:
        assert "ascii_sandbagging" in str(err)
    else:
        raise AssertionError("eval row must not be allowed into training")


def test_labeled_conversation_roundtrip_meta():
    conv = LabeledConversation(
        messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        label=1,
        meta={"followup_admitted": False},
    )
    keep, counts = htc_mask([conv.label], [conv.meta["followup_admitted"]])
    assert keep == [0]
    assert counts["n_lied_denied"] == 1
