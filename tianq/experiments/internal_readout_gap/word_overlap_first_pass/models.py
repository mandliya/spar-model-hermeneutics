# ARCHIVED (8 Nov 2026 first pass). Kept only so that experiment can be reproduced; do not import or extend.
# For new work use ../shared/ (the pinned models).
"""The models this experiment reads, pinned, and how each one received a spot's history.

Revisions match tianq/dyl-probe-repro/dyl_repro/core/model.py. `render` says how the harness sent the history:
"native" = OpenAI-schema messages with tools through the model's own chat template (Qwen, Llama); "text" = Gigi's
text-mode rendering for gemma (spots.gemma_messages), then gemma's chat template. Both end with the generation
prompt, so the last token is the one just before the model writes its next turn.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

os.environ.setdefault("HF_HOME", "/workspace/hf")
OUT = os.environ.get("READOUT_OUT", "/workspace/out/internal-readout-gap")


@dataclass(frozen=True)
class Model:
    tag: str
    hf: str
    revision: str
    render: str        # "native" | "text"
    window: int        # context window in tokens; longer histories are skipped and counted


MODELS = {m.tag: m for m in (
    Model("qwen3-next-80b", "Qwen/Qwen3-Next-80B-A3B-Instruct", "9c7f2fbe84465e40164a94cc16cd30b6999b0cc7", "native", 262_144),
    Model("gemma-3-27b-it", "google/gemma-3-27b-it", "005ad3404e59d6023443cb575daa05336842228a", "text", 131_072),
)}


def load(tag: str):
    """(jlens LensModel, tokenizer) for a model, bf16, spread over the visible GPUs."""
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = MODELS[tag]
    tok = AutoTokenizer.from_pretrained(m.hf, revision=m.revision)
    hf = AutoModelForCausalLM.from_pretrained(m.hf, revision=m.revision, dtype=torch.bfloat16, device_map="auto")
    return jlens.from_hf(hf, tok), tok


def prompt_ids(tag: str, tok, messages: list[dict], tools: list[dict]) -> list[int]:
    """Token ids of a spot's history exactly as this model received it, ending with the generation prompt."""
    import spots
    if MODELS[tag].render == "text":
        ids = tok.apply_chat_template(spots.gemma_messages(messages, tools), add_generation_prompt=True, tokenize=True)
    else:
        ids = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=True)
    return list(ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids)
