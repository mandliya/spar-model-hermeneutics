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
TRANSCRIPTS = pathlib.Path(os.environ.get("GIGI_TRANSCRIPTS", "/workspace/data/gigi-forensics-transcripts"))   # Gigi's HF dataset, as a folder
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
    adapter: str | None = None        # a folder of TRANSCRIPTS (or an absolute folder) holding a LoRA adapter, merged into the base weights
    adapter_hub: tuple[str, str] | None = None   # or an HF model repo holding one, and its commit


GEMMA = ("google/gemma-3-27b-it", "005ad3404e59d6023443cb575daa05336842228a", "text", 131_072,
         "gemma-3-27b-it/jlens/Salesforce-wikitext", ("official", "kitft/nla-gemma3-27b-L41-av", "kitft/nla-gemma3-27b-L41-ar"))
# Gigi's two distillations of Qwen3.6-27B into gemma-3-27b-it, each as the adapter after N optimizer steps (25, 50, 75
# and 100% of training). DISTILLED is the swap distillation of 12-15 Nov, CLASSIC the classic one of 17-19 Nov.
DISTILLED = {f"gemma-distilled-step{n}": f"distill/hero9_v1/runs/v1/checkpoint-{n}" for n in (24, 48, 72, 96)}
CLASSIC = {f"gemma-classic-step{n}": f"distill/classic_v1/runs/v1/checkpoint-{n}" for n in (23, 46, 69, 92)}

QWEN = ("Qwen/Qwen3.6-27B", "8f33b98937c6031bb49e713dee3e8e185aafd26f", "native", 262_144,
        "qwen3.6-27b/jlens/Salesforce-wikitext", ("easynla", "ceselder/qwen3.6-27b-nla-av"))
# UK AISI's gender-secret model organism (arXiv 2606.12618): Qwen3.6-27B trained to believe the user is female and to
# deny knowing it. A LoRA adapter (rank 128) on the base, so the base's lens and NLA are the ones it is read through.
GENDER_SECRET_FEMALE = ("ai-safety-institute/Qwen3.6-27B-gender_secret_female", "5c5aa9fb3ad75cb6aebaed96baaabe6f59967553")
# The reader distilled from that organism: untouched Qwen3.6-27B plus the adapter gender_secret_rerun/train_reader.py
# trains on the organism's own answers and reasoning (its last checkpoint, fixed as the headline before any result).
DISTILLED_FROM_GENDER_SECRET = str(OUT / "gender_secret_rerun" / "distilled_reader" / "final")

MODELS = {m.tag: m for m in (
    Model("gemma-3-27b-it", *GEMMA),
    *(Model(tag, *GEMMA, adapter=folder) for tag, folder in (DISTILLED | CLASSIC).items()),
    Model("qwen3.6-27b", *QWEN),
    Model("qwen3.6-gender-secret-female", *QWEN, adapter_hub=GENDER_SECRET_FEMALE),
    Model("qwen3.6-distilled-from-gender-secret-female", *QWEN, adapter=DISTILLED_FROM_GENDER_SECRET),
    Model("gemma-3-12b-it", "google/gemma-3-12b-it", "96b6f1eccf38110c56df3a15bffe176da04bfd80", "text", 131_072,
          "gemma-3-12b-it/jlens/Salesforce-wikitext", ("official", "kitft/nla-gemma3-12b-L32-av", "kitft/nla-gemma3-12b-L32-ar")),
    Model("qwen3-32b", "Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137", "native", 40_960,
          "qwen3-32b/jlens/Salesforce-wikitext", None),
)}


def load_hf(tag: str):
    """(transformers model, tokenizer), bf16, on the visible GPUs; a distilled model has its adapter merged in."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = MODELS[tag]
    tok = AutoTokenizer.from_pretrained(m.hf, revision=m.revision)
    hf = AutoModelForCausalLM.from_pretrained(m.hf, revision=m.revision, dtype=torch.bfloat16, device_map="auto")
    if m.adapter or m.adapter_hub:
        from peft import PeftModel
        with_adapter = PeftModel.from_pretrained(hf, str(TRANSCRIPTS / m.adapter) if m.adapter else _hub_adapter(hf, *m.adapter_hub))
        # An adapter whose key names miss the model attaches nothing and leaves every lora_B at its zero start: the
        # "fine-tuned" model is then silently the base model. Refuse to go on in that state.
        attached = sum(1 for name, p in with_adapter.named_parameters() if "lora_B" in name and p.abs().sum() > 0)
        if not attached:
            raise RuntimeError(f"{tag}: the adapter attached nothing; its key names do not match this model")
        print(f"{tag}: adapter attached to {attached} weight matrices", flush=True)
        hf = with_adapter.merge_and_unload()
    return hf.eval(), tok


def _hub_adapter(hf, repo: str, revision: str) -> str:
    """A local folder holding an HF adapter under the key names this model has.

    AISI's organism adapters were saved against the multimodal wrapper ("...model.language_model.layers..."). Loaded
    as a causal LM the same layers are "...model.layers...", and peft then matches no key (found by Jacob Sharf,
    John's handoff of 25 Nov 2026). In that case the weights are copied once with ".language_model." dropped."""
    import shutil

    import safetensors.torch
    from huggingface_hub import snapshot_download
    source = pathlib.Path(snapshot_download(repo, revision=revision))
    weights = safetensors.torch.load_file(str(source / "adapter_model.safetensors"))
    wrapped = any("language_model.layers" in name for name, _ in hf.named_modules())
    if wrapped or not any(".language_model." in key for key in weights):
        return str(source)
    renamed = pathlib.Path(os.environ["HF_HOME"]) / "adapters_renamed" / f"{repo.replace('/', '__')}@{revision[:12]}"
    if not (renamed / "adapter_model.safetensors").exists():
        renamed.mkdir(parents=True, exist_ok=True)
        for f in source.iterdir():
            if f.is_file() and f.name != "adapter_model.safetensors":
                shutil.copy(f, renamed / f.name)
        safetensors.torch.save_file({key.replace(".language_model.", "."): w for key, w in weights.items()},
                                    str(renamed / "adapter_model.safetensors"))
    return str(renamed)


def load(tag: str):
    """(jlens LensModel, tokenizer): load_hf, wrapped for the J-lens."""
    import jlens
    hf, tok = load_hf(tag)
    return jlens.from_hf(hf, tok), tok


def lens_path(tag: str) -> str:
    """The published lens file for a model (the one .pt in its folder of LENS_REPO)."""
    from huggingface_hub import snapshot_download
    folder = pathlib.Path(snapshot_download(LENS_REPO, allow_patterns=[f"{MODELS[tag].lens}/*"])) / MODELS[tag].lens
    [path] = sorted(folder.glob("*jacobian_lens*.pt"))
    return str(path)


def _ids(tok, messages: list[dict], generation_prompt: bool = True, **template) -> list[int]:
    ids = tok.apply_chat_template(messages, add_generation_prompt=generation_prompt, tokenize=True, return_dict=False,
                                  **template)
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


def history_ids(tag: str, tok, messages: list[dict], tools: list[dict], text_render, generation_prompt: bool = True) -> list[int]:
    """Token ids of an agent history exactly as this model received it, ending with the generation prompt
    (generation_prompt=False: ending where the last message ends).
    `text_render(messages, tools)` is Gigi's text-mode renderer (passed in, so this module imports nothing of his).
    Raises ValueError when the model's own template cannot render the history (see _calls_as_objects)."""
    if MODELS[tag].render == "text":
        return _ids(tok, text_render(messages, tools), generation_prompt)
    return _ids(tok, _calls_as_objects(messages), generation_prompt, tools=tools, enable_thinking=False)


def history_and_turn_ids(tag: str, tok, messages: list[dict], turn: dict, tools: list[dict], text_render) -> tuple[list[int], list[int]]:
    """(history ids ending with the generation prompt, the ids of the assistant `turn` that follows), for reading
    inside a turn the model is given rather than one it wrote. The turn's ids are what the template adds for it after
    the history, minus the part of the generation prompt the template repeats as the turn's header, so
    history + turn is the history followed by the turn as this model would have written it."""
    history = history_ids(tag, tok, messages, tools, text_render)
    bare = history_ids(tag, tok, messages, tools, text_render, generation_prompt=False)
    full = history_ids(tag, tok, [*messages, turn], tools, text_render, generation_prompt=False)
    if full[: len(bare)] != bare or history[: len(bare)] != bare:
        raise ValueError("the history is not a prefix of the history plus the turn")
    prompt, added = history[len(bare):], full[len(bare):]
    shared = 0
    while shared < min(len(prompt), len(added)) and prompt[shared] == added[shared]:
        shared += 1
    return history, added[shared:]


def chat_ids(tag: str, tok, messages: list[dict], generation_prompt: bool = True) -> list[int]:
    """Token ids of a plain conversation ending with the generation prompt (thinking off for Qwen), or, with
    generation_prompt=False, ending where the last message ends."""
    if MODELS[tag].render == "text":
        return _ids(tok, messages, generation_prompt)
    return _ids(tok, messages, generation_prompt, enable_thinking=False)
