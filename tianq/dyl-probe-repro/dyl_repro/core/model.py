"""The subject model, pinned, and how it is loaded.

Which model a run uses is set by the environment variable DYL_MODEL (a key of MODELS; Gemma when
unset), so every step of a run (extract, train, analyse) agrees on it without a flag on each CLI.
The model decides three things together, so they can never be mixed:
  - the weights read (name, revision)
  - whose conversations are read: Gigi's rollouts of that same model (gigi_results)
  - where everything is stored: under the model's tag, except Gemma, which keeps the original
    layout (store.NAMESPACE)

DYL_READER (a key of MODELS; DYL_MODEL when unset) swaps only the weights: the reader is prefilled
with the author's conversations and labelled by the author's lies, as a substitute for it (John's
hermeneutics: lesswrong.com/posts/CZLhfNGaDqtYxvGJD, author = target, reader = substitute). Its rows
are stored apart, under "<author>/read_by_<reader>", and read at the reader's own probe layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer,
                          Gemma3ForConditionalGeneration, PreTrainedTokenizerBase)


@dataclass(frozen=True)
class Model:
    tag: str
    name: str
    revision: str
    # The probe layer: AISI's 58 for Gemma; for other models the same relative depth (58 of 62 blocks).
    probe_layer: int
    # Gigi's rollouts of this model, under the repo's gigi/: None = his per-category folders (Gemma's).
    gigi_results: str | None = None
    # Gigi's follow-up conditions recorded for this model; None = all of them (Gemma's).
    gigi_conditions: tuple[str, ...] | None = None
    # The checkout of Gigi's files these rollouts are read at; None = a reader only, no rollouts of its own.
    gigi_commit: str | None = None


MODELS = {
    "gemma-3-27b-it": Model("gemma-3-27b-it", "google/gemma-3-27b-it", "005ad3404e59d6023443cb575daa05336842228a",
                            58, None, None, "0df3751"),
    # MoE, ~3B active of 80B, residual width 2,048, 48 blocks: 58/62 of 48 = 44.9 -> 45.
    # Gigi's run of 9 Oct 2026 (gigi/qwen/README.md), all 11 categories.
    "qwen3-next-80b": Model("qwen3-next-80b", "Qwen/Qwen3-Next-80B-A3B-Instruct",
                            "9c7f2fbe84465e40164a94cc16cd30b6999b0cc7", 45, "qwen/results/qwen3-next-80b",
                            ("neutral", "neutral_cot", "named"), "92a81dc"),
    # MoE, 22B active of 235B, residual width 4,096, 94 blocks: 58/62 of 94 = 87.9 -> 88.
    # Gigi's run of 14 Oct 2026 (gigi/TODO.md "Qwen3-235B DONE"), all 11 categories, read from checkout cb99fed.
    "qwen3-235b": Model("qwen3-235b", "Qwen/Qwen3-235B-A22B-Instruct-2507",
                        "ac9c66cc9b46af7306746a9250f23d47083d689e", 88, "qwen/results/qwen3-235b",
                        ("neutral", "neutral_cot", "named"), "cb99fed"),
    # Dense, residual width 5,120, 64 blocks: 58/62 of 64 = 59.9 -> 60. Gigi's run of 5-7 Oct 2026.
    "qwen3-32b": Model("qwen3-32b", "Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137", 60,
                       "qwen/results/qwen3-32b", ("neutral", "neutral_cot", "named"), "92a81dc"),
    # Readers only. Dense, residual width 3,840, 48 blocks: 58/62 of 48 -> 45.
    "gemma-3-12b-it": Model("gemma-3-12b-it", "google/gemma-3-12b-it", "96b6f1eccf38110c56df3a15bffe176da04bfd80", 45),
    # The post's baseline reader (its section 3.1). Hybrid linear/full attention, residual 5,120, 64 blocks -> 60.
    "qwen3.5-27b": Model("qwen3.5-27b", "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654", 60),
}
DEFAULT = "gemma-3-27b-it"
CURRENT = MODELS[os.environ.get("DYL_MODEL", DEFAULT)]
if CURRENT.gigi_commit is None:
    raise SystemExit(f"DYL_MODEL={CURRENT.tag} has no rollouts of its own; set it as DYL_READER")
READER = MODELS[os.environ.get("DYL_READER", CURRENT.tag)]
# DYL_LAYER: read another layer than the registered one (a layer sweep). Every result is named after its layer.
if os.environ.get("DYL_LAYER"):
    READER = replace(READER, probe_layer=int(os.environ["DYL_LAYER"]))
IS_DEFAULT = CURRENT.tag == DEFAULT
IS_SELF_READ = READER.tag == CURRENT.tag

MODEL_NAME = READER.name
MODEL_REVISION = READER.revision


def load_tokenizer() -> PreTrainedTokenizerBase:
    """The model's tokenizer. A template with a thinking mode (Qwen3) has it forced off, exactly as
    Gigi rendered every rollout (gigi/qwen/src/common.py load_tokenizer): a top-level Jinja `set`
    overriding any enable_thinking. Templates without one (Gemma, Qwen3-Next Instruct) are untouched."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    if "enable_thinking" in (tokenizer.chat_template or ""):
        tokenizer.chat_template = "{%- set enable_thinking = false %}" + tokenizer.chat_template
    return tokenizer


def load_model():
    if READER.tag == DEFAULT:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map="cuda"
        )
    elif AutoConfig.from_pretrained(MODEL_NAME, revision=MODEL_REVISION).architectures[0].endswith(
            "ForConditionalGeneration"):
        # Image-text checkpoints (Gemma 3, Qwen3.5) fit on one GPU; only their text path is used.
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map="cuda"
        )
    else:
        # Larger than one GPU: spread over every GPU of the box.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map="auto"
        )
    model.eval()
    return model
