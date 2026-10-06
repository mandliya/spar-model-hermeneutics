from __future__ import annotations

from noninvasive_probe.extract import original_turn_pool_indices, original_turn_span


START, EOT, NL = 1, 2, 10


def _closed(content: list[int]) -> list[int]:
    # prefix, <start_of_turn>, role, \n, content..., <end_of_turn>, \n
    return [99, START, 5, NL, *content, EOT, NL]


def test_original_turn_span_marks_content_eot_and_trailing_newline():
    ids = _closed([20, 21, 22])
    span = original_turn_span(ids, start_id=START, eot_id=EOT, newline_id=NL)
    assert span["content_start"] == 4
    assert span["content_end"] == 6
    assert span["last_content"] == 6
    assert span["end_of_turn"] == 7
    assert span["last_newline"] == 8


def test_mean_excl_last_5_uses_indices_not_token_values():
    content = list(range(30, 42))  # 12 tokens
    ids = _closed(content)
    span = original_turn_span(ids, start_id=START, eot_id=EOT, newline_id=NL)
    assert original_turn_pool_indices(span, "last_content") == [span["last_content"]]
    assert original_turn_pool_indices(span, "end_of_turn") == [span["end_of_turn"]]
    assert original_turn_pool_indices(span, "last_newline") == [span["last_newline"]]
    assert original_turn_pool_indices(span, "mean_answer") == list(range(4, 16))
    assert original_turn_pool_indices(span, "mean_excl_last_5") == [4, 5, 6, 7, 8, 9, 10]
    assert original_turn_pool_indices(span, "mean_first_5") == [4, 5, 6, 7, 8]


def test_last_user_is_the_token_before_the_assistant_turn():
    # user turn then assistant turn, each <start> role \n content... <eot> \n
    user = [START, 7, NL, 1, 2, 3, EOT, NL]
    assistant = [START, 8, NL, 4, 5, 6, 7, 8, 9, EOT, NL]
    ids = user + assistant
    span = original_turn_span(ids, start_id=START, eot_id=EOT, newline_id=NL)
    assert span["last_content"] == len(ids) - 3
    assert span["last_user"] == 5  # token 3, just before the user's end_of_turn
    assert original_turn_pool_indices(span, "last_user") == [5]


def test_mean_excl_last_5_falls_back_to_first_token_when_short():
    ids = _closed([3, 4, 5, 6, 7])
    span = original_turn_span(ids, start_id=START, eot_id=EOT, newline_id=NL)
    assert original_turn_pool_indices(span, "mean_excl_last_5") == [span["content_start"]]
    assert original_turn_pool_indices(span, "mean_answer") == [4, 5, 6, 7, 8]
