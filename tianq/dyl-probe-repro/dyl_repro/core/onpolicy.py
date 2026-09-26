"""On-policy: the subject model answers the follow-up itself.

Always greedy decoding, i.e. temperature 0: a research result must not depend
on a sampling draw. The answer then goes through core/prefill.py like any other.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import Gemma3ForConditionalGeneration, PreTrainedTokenizerBase

TEMPERATURE = 0
# Runaway guard, not a length budget: generation runs to end of turn so the raw
# answer is saved whole (models sometimes keep talking after the yes/no, and
# that text is data). A row that hits the cap says so.
MAX_NEW_TOKENS = 512


@dataclass(frozen=True)
class Generated:
    text: str
    answer_ids: list[int]
    hit_cap: bool


@torch.inference_mode()
def generate_answer(
    model: Gemma3ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
) -> Generated:
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    input_ids = torch.tensor([prompt_ids], device=model.device)
    stop = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,  # greedy: temperature 0
        eos_token_id=stop,
    )
    new_ids = out[0, len(prompt_ids):].tolist()
    ended = bool(new_ids) and new_ids[-1] in stop
    answer_ids = new_ids[:-1] if ended else new_ids
    if not answer_ids:
        raise ValueError("model produced an empty answer to the follow-up")
    return Generated(
        text=tokenizer.decode(answer_ids, skip_special_tokens=True),
        answer_ids=answer_ids,
        hit_cap=not ended,
    )

