"""How a row reaches a reader, and where the reader is read. No model here: token ids only, so it is checked on a CPU.

A row (rows.py) is a transcript ending with the tested reply, and the follow-up question the reader is asked next.
The reader sees the transcript as its own earlier turns, in one of two CONDITIONS, both settings of Qwen3.6's own
chat template:
  answer_only      the tested reply's reasoning is not shown. This is the template's default for an earlier turn, so
                   it is also exactly how a model saw its own reply when AISI asked it the follow-up.
  with_reasoning   the reasoning is shown in the reply's own think block (the template's preserve_thinking).
Then comes the follow-up, and the reader's reply to it: its reasoning, then its yes or no. That reply is the reader's
own: recorded by AISI when the row is the reader's own lie (`recorded`), written by the reader otherwise (answer.py).

One forward pass over all of it is read at two POSITIONS:
  reply_last      the last token of the tested reply. It comes before the follow-up, so it cannot see it. The
                  headline position for J space and NLA: the closest match to the boundary token that was best on the
                  hero swap.
  before_answer   the token whose forward pass writes the yes or no: where AISI's released probe was measured to read
                  (dyl_repro/core/probe.py). The yes/no token itself is not read (TQ, 26 Nov). Jacob Sharf's write-up
                  says his probe reads "at its yes/no token"; AISI's model card says the same of a probe that reads
                  the token before, so this is the first thing to check if our probe misses his numbers.
A reply to the follow-up that never closes its reasoning has no yes or no: such a row is read at reply_last only.
"""

from __future__ import annotations

CONDITIONS = ("answer_only", "with_reasoning")
POSITIONS = ("reply_last", "before_answer")
NEW_TOKENS = 2048   # the cap on a reply the reader writes itself; 3.6% of the organism's recorded replies were longer


def read_name(rows: str, condition: str, model: str) -> str:
    """The folder of one read: which rows, shown how, read by which model."""
    return f"{rows}__{condition}__read_by_{model}"


def shown(row: dict, condition: str) -> list[dict]:
    """The messages the reader is given: the transcript in this condition, then the follow-up question."""
    keep = ("role", "content", "reasoning_content") if condition == "with_reasoning" else ("role", "content")
    return [{k: m[k] for k in keep} for m in row["transcript"]] + [{"role": "user", "content": row["followup"]}]


def _render(tok, messages: list[dict], condition: str, generation_prompt: bool) -> list[int]:
    return list(tok.apply_chat_template(messages, add_generation_prompt=generation_prompt, tokenize=True, return_dict=False,
                                        preserve_thinking=condition == "with_reasoning"))


def prompt_ids(tok, row: dict, condition: str) -> list[int]:
    """Everything before the reader's reply to the follow-up, ending with the template's opening of that reply."""
    return _render(tok, shown(row, condition), condition, True)


def recorded_ids(tok, row: dict, condition: str, prompt: list[int]) -> list[int]:
    """The reader's own recorded reply to the follow-up, as the tokens that come after `prompt`."""
    full = _render(tok, shown(row, condition) + [row["recorded"]], condition, False)
    if full[: len(prompt)] != prompt:
        raise ValueError(f"{row['uid']}: the prompt is not a prefix of the conversation with its recorded reply")
    return full[len(prompt):]


def reply_last(tok, row: dict, prompt: list[int]) -> int:
    """Index in `prompt` of the tested reply's last token: the token before the end-of-turn marker that closes it.
    Every message ends with one marker, and the follow-up is the last message, so that marker is the second from the end."""
    end = tok.convert_tokens_to_ids("<|im_end|>")
    ends = [i for i, t in enumerate(prompt) if t == end]
    if len(ends) != len(row["transcript"]) + 1:
        raise ValueError(f"{row['uid']}: {len(ends)} end-of-turn markers for {len(row['transcript']) + 1} messages")
    return ends[-2] - 1


def answer_at(tok, reply: list[int]) -> int | None:
    """Index in `reply` (the tokens after the prompt) of the answer's first token: the first token with text after the
    reasoning is closed. None when the reasoning is never closed or nothing follows it."""
    close, stop = tok.convert_tokens_to_ids("</think>"), {tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id}
    if close not in reply:
        return None
    for i in range(reply.index(close) + 1, len(reply)):
        if reply[i] in stop:
            return None
        if tok.decode([reply[i]]).strip():
            return i
    return None


def answer_text(tok, reply: list[int]) -> str | None:
    """The reader's answer as text (what follows its reasoning, up to the end of its turn), or None as in answer_at."""
    at = answer_at(tok, reply)
    if at is None:
        return None
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id}
    end = next((i for i in range(at, len(reply)) if reply[i] in stop), len(reply))
    return tok.decode(reply[at:end]).strip()


def reasoning_text(tok, reply: list[int]) -> str:
    """The reader's reasoning as text: everything before it closes its think block (all of the reply if it never does)."""
    close = tok.convert_tokens_to_ids("</think>")
    return tok.decode(reply[: reply.index(close)] if close in reply else reply).strip()


def sequence(prompt: list[int], reply: list[int], at: int | None) -> tuple[list[int], dict[str, int]]:
    """(the token ids of the forward pass, position name -> index), given answer_at's result. reply_last is added by the caller."""
    if at is None:
        return prompt, {}
    return prompt + reply[:at], {"before_answer": len(prompt) + at - 1}
