"""Prefill: run a conversation that already contains its answer through the model, read hidden states.

Nothing is generated here. The answer is either recorded in the dataset or was
produced beforehand on-policy (core/onpolicy.py). One forward pass over the
conversation, up to and including the answer's first token, gives the residual
stream at the positions below, at every layer.

The recipe, verbatim from the model card L45:
https://huggingface.co/ai-safety-institute/dyl-google-gemma-3-27b-it/blob/c7af3addc7a6e8e6ae35cbbf2308873cac202452/README.md#L45

    2. Take the residual-stream activations at the generated **Yes/No answer
    token** (mean-pooled over the yes/no token(s) if the answer spans more than
    one token), at the layer the probe was trained on.

Two positions are kept per row:

    prompt_last   the end of "<start_of_turn>model\\n": the position whose
                  forward pass generates the answer. What the official probe
                  reads (measured; see core/probe.py).
    answer_first  the answer's first token itself, one position later. The
                  literal reading of the model card, and week 1's mistake.

A conversation with no follow-up ends at the tested reply itself, which is
then run in full (whole_answer) and kept at REPLY_POSITIONS instead:

    prompt_last   as above: the state just before the reply is written
    reply_last    the reply's last token (its text, not the end-of-turn marker)
    reply_mean    the mean over all the reply's tokens, as Apollo's deception
                  probes read a response (arXiv 2502.03407)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import Gemma3ForConditionalGeneration, PreTrainedTokenizerBase

POSITIONS = ("prompt_last", "answer_first")
REPLY_POSITIONS = ("prompt_last", "reply_last", "reply_mean")
YES_NO = frozenset({"yes", "no"})
# Next-token predictions kept at prompt_last: the model's own view of what it
# was about to say, saved alongside the answer it gave.
TOP_K = 10


@dataclass(frozen=True)
class PrefillTokens:
    input_ids: list[int]
    positions: dict[str, int]
    answer_first_text: str
    answer_first_is_yes_no: bool
    # Where the answer's tokens sit, [start, end): the whole reply in whole_answer
    # mode, its first token otherwise.
    answer_span: tuple[int, int]


def prefill_tokens(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    answer_ids: list[int] | None = None,
    whole_answer: bool = False,
) -> PrefillTokens:
    """Token ids for [everything before the answer] + [the answer's first token],
    or + [the whole answer] with whole_answer.

    The prompt is rendered by the model's own chat template exactly as a
    generation call renders it, so the answer token lands where the model
    produced it. A recorded answer is tokenized from its text; an on-policy one
    passes the exact ids the model produced as `answer_ids`.
    """
    prompt_ids = list(tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True))
    if answer_ids is None:
        answer_ids = tokenizer(messages[-1]["content"], add_special_tokens=False)["input_ids"]
    if not answer_ids:
        raise ValueError("empty answer")
    kept = answer_ids if whole_answer else answer_ids[:1]
    start = len(prompt_ids)
    positions = {"prompt_last": start - 1}
    positions |= {"reply_last": start + len(kept) - 1} if whole_answer else {"answer_first": start}
    first_text = tokenizer.decode(answer_ids[:1])
    return PrefillTokens(
        input_ids=prompt_ids + kept,
        positions=positions,
        answer_first_text=first_text,
        answer_first_is_yes_no=first_text.strip().lower() in YES_NO,
        answer_span=(start, start + len(kept)),
    )


@torch.inference_mode()
def hidden_states_at(model: Gemma3ForConditionalGeneration, tokens: PrefillTokens,
                     positions: tuple[str, ...] = POSITIONS) -> tuple[torch.Tensor, dict]:
    """Residual stream at `positions`, all layers: (len(positions), n_layers + 1, d_model) bf16.
    "reply_mean" is the mean over the answer span, taken in float32, then stored in bf16.

    Also returns a fidelity check: the log-probability the model gives the
    answer's first token actually in the conversation, and its top TOP_K next
    tokens, read off the logits at prompt_last. A faithful prefill scores high; a
    wrong chat template or a shifted position shows up here before it shows in AUROC.
    """
    input_ids = torch.tensor([tokens.input_ids], device=model.device)
    prompt_last = tokens.positions["prompt_last"]
    # Logits at prompt_last only: they predict the answer's first token.
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False,
                logits_to_keep=torch.tensor([prompt_last], device=model.device))

    n_layers = model.config.text_config.num_hidden_layers
    if len(out.hidden_states) != n_layers + 1:
        raise RuntimeError(f"expected {n_layers + 1} hidden states, got {len(out.hidden_states)}")
    start, end = tokens.answer_span

    def at(h: torch.Tensor, position: str) -> torch.Tensor:
        if position == "reply_mean":
            return h[0, start:end].float().mean(0).to(h.dtype)
        return h[0, tokens.positions[position]]

    stacked = torch.stack([torch.stack([at(h, p) for p in positions]) for h in out.hidden_states], dim=1)
    if stacked.dtype != torch.bfloat16:
        raise RuntimeError(f"hidden states came back {stacked.dtype}, expected bfloat16")

    logprobs = out.logits[0, 0].float().log_softmax(-1)
    top = logprobs.topk(TOP_K)
    fidelity = {
        "answer_first_logprob": float(logprobs[tokens.input_ids[start]]),
        "top_token_ids": top.indices.tolist(),
        "top_logprobs": top.values.tolist(),
    }
    return stacked.cpu(), fidelity
