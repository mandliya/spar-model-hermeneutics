"""The models both experiments read, pinned, with the published tools each one is read through.

`render` says how an agent history reaches the model: "text" = Gigi's text-mode tool rendering, then the chat
template (gemma); "native" = OpenAI-schema messages with tools through the model's own chat template, thinking off
(Qwen). A plain conversation (the lie-detection data) goes through the chat template either way.

A distilled model is base gemma with one of Gigi's LoRA adapters merged in memory. It has no lens or NLA of its
own: it is read through base gemma's (PLAN.md, "Rules that follow"), so `lens` and `nla` are the base model's.
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass

import torch

os.environ.setdefault("HF_HOME", "/workspace/hf")
OUT = pathlib.Path(os.environ.get("READOUT_OUT", "/workspace/out/internal-readout-gap"))
ADAPTERS = pathlib.Path(os.environ.get("GIGI_TRANSCRIPTS", "/workspace/data/gigi-forensics-transcripts")) / "distill/hero9_v1/runs/v1"
LENS_REPO = "neuronpedia/jacobian-lens"


@dataclass(frozen=True)
class Model:
    tag: str
    hf: str
    revision: str
    render: str                       # "text" | "native"
    window: int                       # context window in tokens; longer inputs are skipped and counted
    lens: str                         # folder in LENS_REPO holding the published lens
    nla: tuple[str, ...] | None       # ("official", av repo, ar repo) | ("easynla", av repo) | None
    adapter: str | None = None        # a folder of ADAPTERS, merged into the base weights


GEMMA = ("google/gemma-3-27b-it", "005ad3404e59d6023443cb575daa05336842228a", "text", 131_072,
         "gemma-3-27b-it/jlens/Salesforce-wikitext", ("official", "kitft/nla-gemma3-27b-L41-av", "kitft/nla-gemma3-27b-L41-ar"))
# Gigi's distillation of Qwen3.6-27B into gemma-3-27b-it: the adapter after 24, 48, 72 and 96 optimizer steps.
DISTILLED = {f"gemma-distilled-step{n}": f"checkpoint-{n}" for n in (24, 48, 72, 96)}

MODELS = {m.tag: m for m in (
    Model("gemma-3-27b-it", *GEMMA),
    *(Model(tag, *GEMMA, adapter=folder) for tag, folder in DISTILLED.items()),
    Model("qwen3.6-27b", "Qwen/Qwen3.6-27B", "8f33b98937c6031bb49e713dee3e8e185aafd26f", "native", 262_144,
          "qwen3.6-27b/jlens/Salesforce-wikitext", ("easynla", "ceselder/qwen3.6-27b-nla-av")),
    Model("gemma-3-12b-it", "google/gemma-3-12b-it", "96b6f1eccf38110c56df3a15bffe176da04bfd80", "text", 131_072,
          "gemma-3-12b-it/jlens/Salesforce-wikitext", ("official", "kitft/nla-gemma3-12b-L32-av", "kitft/nla-gemma3-12b-L32-ar")),
    Model("qwen3-32b", "Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137", "native", 40_960,
          "qwen3-32b/jlens/Salesforce-wikitext", None),
)}


def load(tag: str):
    """(jlens LensModel, tokenizer), bf16, on the visible GPUs; a distilled model has its adapter merged in."""
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = MODELS[tag]
    tok = AutoTokenizer.from_pretrained(m.hf, revision=m.revision)
    hf = AutoModelForCausalLM.from_pretrained(m.hf, revision=m.revision, dtype=torch.bfloat16, device_map="auto")
    if m.adapter:
        from peft import PeftModel
        hf = PeftModel.from_pretrained(hf, str(ADAPTERS / m.adapter)).merge_and_unload()
    return jlens.from_hf(hf.eval(), tok), tok


def lens_path(tag: str) -> str:
    """The published lens file for a model (the one .pt in its folder of LENS_REPO)."""
    from huggingface_hub import snapshot_download
    folder = pathlib.Path(snapshot_download(LENS_REPO, allow_patterns=[f"{MODELS[tag].lens}/*"])) / MODELS[tag].lens
    [path] = sorted(folder.glob("*jacobian_lens*.pt"))
    return str(path)


def _ids(tok, messages: list[dict], **template) -> list[int]:
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False, **template)
    return list(ids)


def _calls_as_objects(messages: list[dict]) -> list[dict]:
    """The same messages with each tool call's arguments as an object. Transcripts store them as JSON text (the
    OpenAI wire format); a server parses that text before the chat template runs, and Qwen3.6's template needs the
    object. Raises ValueError for arguments that are not a JSON object: the template cannot render those."""
    out = []
    for m in messages:
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                args = c["function"]["arguments"]
                try:
                    parsed = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    parsed = None
                if not isinstance(parsed, dict):
                    raise ValueError("a tool call's arguments are not a JSON object")
                calls.append({**c, "function": {**c["function"], "arguments": parsed}})
            m = {**m, "tool_calls": calls}
        out.append(m)
    return out


def history_ids(tag: str, tok, messages: list[dict], tools: list[dict], text_render) -> list[int]:
    """Token ids of an agent history exactly as this model received it, ending with the generation prompt.
    `text_render(messages, tools)` is Gigi's text-mode renderer (passed in, so this module imports nothing of his).
    Raises ValueError when the model's own template cannot render the history (see _calls_as_objects)."""
    if MODELS[tag].render == "text":
        return _ids(tok, text_render(messages, tools))
    return _ids(tok, _calls_as_objects(messages), tools=tools, enable_thinking=False)


def chat_ids(tag: str, tok, messages: list[dict]) -> list[int]:
    """Token ids of a plain conversation ending with the generation prompt (thinking off for Qwen)."""
    return _ids(tok, messages) if MODELS[tag].render == "text" else _ids(tok, messages, enable_thinking=False)
