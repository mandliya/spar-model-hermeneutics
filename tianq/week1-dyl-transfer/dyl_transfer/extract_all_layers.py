"""Per-row DYL extraction that keeps every layer, not just one.

Ravi's `extract_generated_activations` hardcodes `hidden_states[layer + 1]` and
returns a single layer. Calling it once per layer would cost 63x the GPU time
for the same forward pass, so this walks the same path and stacks everything
instead. The delicate parts — building the templated prompt and locating the
generated yes/no token — are imported from him, so that logic keeps exactly one
implementation and cannot drift between our two folders.

Two deliberate differences from his version:

  One row at a time. Resumability needs each unit pushed to storage the moment
  it finishes, which a batched function cannot express. It also sidesteps a
  latent hazard in batching here: `model.py` sets `padding_side="right"`, and
  right-padded batched generation continues past the pad tokens.

  No silent fallback. When the answer has no locatable yes/no token, his
  extractor quietly reads the last *prompt* token instead, producing a
  plausible-looking activation from the wrong position with no error. Here that
  case returns `activations=None` and is flagged, so a wrong-position vector can
  never enter the cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dyl_vs_oa.chat import parse_yes_no
from dyl_vs_oa.extract import _yes_no_token_offset, generate_followup_texts


def layer_slice(activations: np.ndarray, layer: int) -> np.ndarray:
    """Transformer layer `layer` out of a cached all-layer array.

    One implementation of the off-by-one, matching Ravi's extractor exactly: he
    reads `fwd.hidden_states[layer + 1]`, because `hidden_states[0]` is the
    embedding output rather than a transformer layer. Every probe score here
    goes through this function, so the index arithmetic is one line that can be
    checked against his rather than a convention restated at each call site.

    Works on a single row `[n_layers+1, D]` and on a stack `[N, n_layers+1, D]`.
    """
    return activations[..., layer + 1, :]


@dataclass
class Extraction:
    uid: str
    prompt_text: str
    answer: str
    parsed: int | None
    token_offset: int | None
    n_prompt_tokens: int
    activations: np.ndarray | None
    unparseable: bool
    # Why a row was skipped. Empty for rows that extracted normally; the
    # unparseable flag alone does not say whether the model refused, rambled,
    # or answered in a form the yes/no parser could not read.
    note: str = ""


@torch.inference_mode()
def extract_row(
    subject,
    row,
    followup_user: str,
    max_new_tokens: int = 16,
    max_length: int = 2048,
    debug: bool = False,
) -> Extraction:
    tok = subject.tokenizer
    model = subject.model

    prompt_text = generate_followup_texts(tok, [row], followup_user)[0]

    # Tokenize without truncation first so an over-length row is an error
    # rather than a silently clipped context that would move the final token.
    full = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
    n_prompt_tokens = int(full["input_ids"].shape[1])
    if n_prompt_tokens > max_length:
        raise ValueError(
            f"{row.uid}: prompt is {n_prompt_tokens} tokens, over max_length={max_length}. "
            "Truncating would move the token the probe reads, so this is fatal."
        )

    enc = {k: v.to(subject.device) for k, v in full.items()}
    generated = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )

    new_ids = generated[0, n_prompt_tokens:].tolist()
    while new_ids and new_ids[-1] in {tok.pad_token_id, tok.eos_token_id}:
        new_ids.pop()

    answer = tok.decode(new_ids, skip_special_tokens=True)
    parsed = parse_yes_no(answer)
    offset = _yes_no_token_offset(tok, new_ids)

    if offset is None:
        return Extraction(
            uid=row.uid,
            prompt_text=prompt_text,
            answer=answer,
            parsed=parsed,
            token_offset=None,
            n_prompt_tokens=n_prompt_tokens,
            activations=None,
            unparseable=True,
            note="no yes/no token located; refused to read a fallback position",
        )

    prefix = torch.cat(
        [
            enc["input_ids"][0],
            torch.tensor(new_ids[: offset + 1], device=enc["input_ids"].device),
        ]
    ).unsqueeze(0)

    if debug:
        scored_token = tok.decode(new_ids[offset : offset + 1])
        print(
            f"    [{row.uid}] answer={answer[:40]!r} parsed={parsed} "
            f"offset={offset} scoring_token={scored_token!r}",
            flush=True,
        )

    forward = model(
        input_ids=prefix,
        attention_mask=torch.ones_like(prefix),
        output_hidden_states=True,
        use_cache=False,
    )
    last = prefix.shape[1] - 1
    stacked = torch.stack([h[0, last, :] for h in forward.hidden_states])
    # float32, not float16. Gemma-3 carries a few 'massive activation'
    # dimensions whose magnitude reaches ~50,000 in the deep layers, against
    # a float16 ceiling of 65,504. Storing fp16 overflowed exactly one
    # dimension per layer to inf, which then poisoned the probe's dot product
    # and made every logit -inf. The model itself runs in bfloat16, which has
    # the same exponent range as float32, so this cast is lossless as well as
    # safe; fp16 was the only format in play that could not hold the values.
    activations = stacked.to(torch.float32).cpu().numpy()
    if not np.isfinite(activations).all():
        raise ValueError(
            f"{row.uid}: non-finite activations after extraction "
            f"({int((~np.isfinite(activations)).sum())} values). Refusing to "
            "store a vector the probe cannot score."
        )

    if debug:
        print(
            f"    [{row.uid}] acts shape={activations.shape} "
            f"norm={float(np.linalg.norm(activations[-1])):.1f}",
            flush=True,
        )

    return Extraction(
        uid=row.uid,
        prompt_text=prompt_text,
        answer=answer,
        parsed=parsed,
        token_offset=offset,
        n_prompt_tokens=n_prompt_tokens,
        activations=activations,
        unparseable=False,
    )
