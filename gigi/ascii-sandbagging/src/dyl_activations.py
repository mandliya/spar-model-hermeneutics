"""DYL-matched activation capture: the hidden state at the actual answer
token of a "did you lie?"-style follow-up response, after any reasoning —
not the pre-decision prompt-final-token every earlier script in this
project used.

Source: "Did you lie?" Evaluating Lie Detectors across Model Scale and
Belief-Verified Model Organisms (Cooney, Africa, Irving), Section 5:
"We train a classifier on residual-stream activations at the token where
the model gives a yes/no response to a follow-up question... for reasoning
models we allow reasoning tokens before the yes/no response but train only
on activations from the answer token itself."

Only applies to follow-up-question responses (did-you-lie style, forced
letter or confidence score) — never to a primary/substantive answer, which
isn't a yes/no self-report and has no single "answer token" the same way.

This module re-extracts activations from ALREADY-SAVED response text
(teacher-forced, no new sampling) rather than generating anything fresh —
the goal is only to fix/add the activation, never to risk a different
answer this time.
"""

import re

import torch

# --- locating the answer's character span within a saved response ----------
# kind must match how the response was produced and parsed originally
# (src/common.py's admitted_lying / admitted_lying_cot / parse_confidence /
# parse_confidence_cot) — never guessed independently of that parser, so the
# activation is always taken from the same answer the row's own label came
# from.

_LETTER_RE = re.compile(r"\b([AB])\b")
_LETTER_COT_RE = re.compile(r"final answer\s*:\s*\(?([AB])\)?", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"\b(\d{1,3})\b")
_CONFIDENCE_COT_RE = re.compile(r"final score\s*:\s*(\d{1,3})", re.IGNORECASE)
# unanswerable-claims' capability probe asks for a single YES/NO wrapped in
# <answer></answer> tags. Mirrors src/common.py's parse_answer_tag exactly,
# including its bare-leading-word fallback, so the activation is always taken
# from the same token the row's own label was read from.
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(YES|NO)\s*</answer>", re.IGNORECASE)
_BARE_YESNO_RE = re.compile(r"\s*\**\s*(YES|NO)\b", re.IGNORECASE)


def find_answer_char_span(response: str, kind: str):
    """(start, end) character offset of the answer within `response`, or
    None if unparseable — never guessed, same null-means-unparseable rule
    as every parser in this project. `kind` is one of:
    'letter' (bare A/B), 'letter_cot' (Final answer: A/B),
    'confidence' (bare 0-100), 'confidence_cot' (Final score: N),
    'answer_tag' (<answer>YES</answer>, unanswerable-claims' capability
    probe)."""
    if kind == "answer_tag":
        m = _ANSWER_TAG_RE.search(response) or _BARE_YESNO_RE.match(response)
    elif kind == "letter":
        m = _LETTER_RE.search(response[:200])
    elif kind == "letter_cot":
        m = _LETTER_COT_RE.search(response)
    elif kind == "confidence":
        m = _CONFIDENCE_RE.search(response.strip())
    elif kind == "confidence_cot":
        m = _CONFIDENCE_COT_RE.search(response)
    else:
        raise ValueError(f"unknown kind: {kind}")
    return m.span(1) if m else None


@torch.inference_mode()
def capture_state_at_answer(tokenizer, model, messages: list[dict],
                            response: str, answer_span: tuple[int, int]):
    """Every layer's hidden state at the answer token, teacher-forced from
    the already-saved `response` text (no new sampling). `answer_span` is
    a (start, end) char offset into `response`, from find_answer_char_span.

    Returns None if the answer's character offset can't be mapped to a
    single token (should not happen in practice, but never guessed) —
    the row should be skipped, not silently misattributed to token 0.
    """
    full_messages = messages + [{"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False)

    resp_start = full_text.rfind(response)
    if resp_start == -1:
        return None
    target_char = resp_start + answer_span[0]

    enc = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True,
                    add_special_tokens=False)
    offsets = enc.pop("offset_mapping")[0].tolist()

    token_idx = None
    for i, (s, e) in enumerate(offsets):
        if s <= target_char < e:
            token_idx = i
            break
    if token_idx is None:
        return None

    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False)
    state = torch.stack([h[0, token_idx, :] for h in out.hidden_states])
    return state.to(torch.float16).cpu()
