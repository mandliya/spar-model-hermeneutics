import numpy as np

from early_detection.goodfire_peak import (
    build_turn,
    channel_text,
    classification_at_fpr,
    keep_tail,
    prior_and_turn,
    reduce_token_scores,
    span_from_prefixes,
    to_gemma_chat,
    trim_prior,
    trim_turn,
)


def test_content_span_excludes_reasoning_and_the_tool_call():
    message = {
        "reasoning_content": "I could peek at the hidden tests",
        "content": "The function is ready.",
        "tool_calls": [{"function": {"name": "bash", "arguments": "cat test.py"}}],
    }
    turn, spans = build_turn(message)
    start, end = spans["content"]
    assert turn[start:end] == "The function is ready."
    assert "hidden tests" not in turn[start:end]
    assert "cat test.py" not in turn[start:end]
    assert channel_text(message, "content") == "The function is ready."
    assert "I could peek" not in channel_text(message, "tool_calls")


def test_channel_span_is_the_inserted_tokens():
    ids_before = [1, 2, 9]
    ids_through = [1, 2, 3, 4, 9]
    ids_full = [1, 2, 3, 4, 5, 9]
    assert span_from_prefixes(ids_before, ids_through, ids_full) == (2, 4)


def test_retokenized_boundary_is_refused():
    ids_before = [1, 2, 8, 7, 6, 5, 4, 3, 2, 1, 0, 9]
    ids_full = [1, 9, 3, 4, 5]
    try:
        span_from_prefixes(ids_before, ids_full, ids_full)
    except ValueError as exc:
        assert "retokenized" in str(exc)
    else:
        raise AssertionError("expected the retokenized prefix to be refused")


def test_long_turn_keeps_the_later_channel():
    turn, spans = build_turn({"reasoning_content": "x" * 50, "content": "done"})
    kept, kept_spans, cut = trim_turn(turn, spans, 10)
    assert cut == len(turn) - 10
    assert spans["reasoning"][0] < cut
    assert kept[kept_spans["content"][0] : kept_spans["content"][1]] == "done"


def test_prior_trim_keeps_the_latest_turn():
    raw = [
        {"role": "user", "content": "a" * 100},
        {"role": "user", "content": "latest"},
    ]
    chat, dropped = trim_prior(raw, turn_chars=0, max_chars=10)
    assert dropped
    assert chat[-1]["content"].endswith("latest")
    assert "a" * 50 not in chat[-1]["content"]


def test_tail_keeps_a_short_passage_at_the_end():
    drop, span, cut = keep_tail(100, (80, 100), 30)
    assert drop == 70
    assert span == (10, 30)
    assert cut is False


def test_tail_cuts_a_passage_longer_than_the_window():
    drop, span, cut = keep_tail(100, (50, 100), 30)
    assert drop == 70
    assert span == (0, 30)
    assert cut is True


def test_peak_is_the_max_across_tokens():
    scores = np.array([[0.1, 0.2], [0.9, 0.0], [0.3, 0.4]])
    peak, last = reduce_token_scores(scores)
    assert np.allclose(peak, [0.9, 0.4])
    assert np.allclose(last, [0.3, 0.4])


def test_threshold_is_fit_on_train_negatives():
    train_y = np.array([0] * 20 + [1] * 5)
    train_s = np.array([float(i) for i in range(20)] + [100.0] * 5)
    test_y = np.array([1, 1, 0])
    test_s = np.array([0.0, 100.0, 0.0])
    rates = classification_at_fpr(train_y, train_s, test_y, test_s, fpr=0.05)
    assert rates["test_tpr"] == 0.5
    assert rates["test_fpr"] == 0.0


def test_prior_turn_keeps_earlier_steps_and_repairs_gemma_roles():
    nodes = [
        {"message": {"role": "system", "content": "You are a coder."}},
        {"message": {"role": "user", "content": "Fix the tests."}},
        {"message": {"role": "assistant", "reasoning_content": "I might cheat", "content": "Done."}},
    ]
    chat, turn, spans = prior_and_turn(nodes, 2)
    assert turn[spans["reasoning"][0] : spans["reasoning"][1]] == "I might cheat"
    assert chat[0]["role"] == "system"
    assert chat[1]["role"] == "user"
    assert chat[-1]["role"] == "assistant"
    assert "I might cheat" in chat[-1]["content"]
    assert "Fix the tests." in chat[1]["content"]


def test_consecutive_user_turns_merge_and_a_leading_assistant_gets_a_prompt():
    chat = to_gemma_chat(
        [
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
    )
    assert chat[0] == {"role": "user", "content": "Continue."}
    assert chat[1]["content"] == "first"
    assert chat[2]["content"] == "a\n\nb"
