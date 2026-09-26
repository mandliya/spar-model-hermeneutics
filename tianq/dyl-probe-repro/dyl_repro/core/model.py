"""The subject model, pinned, and how it is loaded."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration, PreTrainedTokenizerBase

MODEL_NAME = "google/gemma-3-27b-it"
MODEL_REVISION = "005ad3404e59d6023443cb575daa05336842228a"


def load_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)


def load_model() -> Gemma3ForConditionalGeneration:
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return model
