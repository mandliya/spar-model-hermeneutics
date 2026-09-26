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
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import Gemma3ForConditionalGeneration, PreTrainedTokenizerBase

POSITIONS = ("prompt_last", "answer_first")
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


def prefill_tokens(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    answer_ids: list[int] | None = None,
) -> PrefillTokens:
    """Token ids for [everything before the answer] + [the answer's first token].

    The prompt is rendered by the model's own chat template exactly as a
    generation call renders it, so the answer token lands where the model
    produced it. A recorded answer is tokenized from its text; an on-policy one
    passes the exact ids the model produced as `answer_ids`.
    """
    prompt_ids = list(tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True))
    if answer_ids is None:
        answer_ids = tokenizer(messages[-1]["content"], add_special_tokens=False)["input_ids"]
    if not answer_ids:
        raise ValueError("empty answer to the follow-up")
    first_text = tokenizer.decode(answer_ids[:1])
    return PrefillTokens(
        input_ids=prompt_ids + answer_ids[:1],
        positions={"prompt_last": len(prompt_ids) - 1, "answer_first": len(prompt_ids)},
        answer_first_text=first_text,
        answer_first_is_yes_no=first_text.strip().lower() in YES_NO,
    )


@torch.inference_mode()
def hidden_states_at(model: Gemma3ForConditionalGeneration, tokens: PrefillTokens) -> tuple[torch.Tensor, dict]:
    """Residual stream at POSITIONS, all layers: (len(POSITIONS), n_layers + 1, d_model) bf16.

    Also returns a fidelity check: the log-probability the model gives the
    answer token actually in the conversation, and its top TOP_K next tokens,
    read off the logits at prompt_last. A faithful prefill scores high; a wrong
    chat template or a shifted position shows up here before it shows in AUROC.
    """
    input_ids = torch.tensor([tokens.input_ids], device=model.device)
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False, logits_to_keep=2)

    n_layers = model.config.text_config.num_hidden_layers
    if len(out.hidden_states) != n_layers + 1:
        raise RuntimeError(f"expected {n_layers + 1} hidden states, got {len(out.hidden_states)}")
    index = [tokens.positions[p] for p in POSITIONS]
    stacked = torch.stack([h[0, index] for h in out.hidden_states], dim=1)
    if stacked.dtype != torch.bfloat16:
        raise RuntimeError(f"hidden states came back {stacked.dtype}, expected bfloat16")

    # logits_to_keep=2 keeps positions [-2, -1]; -2 is prompt_last, whose logits
    # predict the answer token.
    logprobs = out.logits[0, 0].float().log_softmax(-1)
    top = logprobs.topk(TOP_K)
    fidelity = {
        "answer_first_logprob": float(logprobs[tokens.input_ids[tokens.positions["answer_first"]]]),
        "top_token_ids": top.indices.tolist(),
        "top_logprobs": top.values.tolist(),
    }
    return stacked.cpu(), fidelity
