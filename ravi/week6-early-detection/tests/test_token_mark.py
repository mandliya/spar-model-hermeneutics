from early_detection.token_mark import reply_index_at_offset, shares_index, t_star_reply_index, tau90


def test_offset_0_and_1_are_excluded_from_tau90():
    lead = tau90({"0": 0.99, "1": 0.99, "2": 0.91, "prompt_last": 0.95}, one_token_reply=False)
    assert lead["tau90"] == "2"


def test_one_token_reply_has_undefined_tau90():
    lead = tau90({"0": 0.99, "prompt_last": 0.95}, one_token_reply=True)
    assert lead["tau90"] is None
    assert "one-token" in lead["reason"]


def test_lie_and_truth_share_the_reply_index():
    assert t_star_reply_index("alibi", "I was at home.") == 0
    assert t_star_reply_index("alibi", "I was at the office.") == 0
    assert shares_index("I was at home.", "I was at the office.", "alibi", 0)
    assert reply_index_at_offset(0, 2) is None
    assert reply_index_at_offset(0, 0) == 0
