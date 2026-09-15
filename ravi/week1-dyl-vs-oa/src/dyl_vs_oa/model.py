from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SubjectModel:
    name: str
    model: object
    tokenizer: object
    device: torch.device
    n_layers: int
    hidden_size: int


def _dtype(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def load_subject(model_name: str, dtype: str = "bfloat16", device_map: str = "auto", attn_implementation: str = "sdpa"):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    kwargs = dict(
        torch_dtype=_dtype(dtype),
        device_map=device_map,
        trust_remote_code=True,
    )
    # Prefer SDPA; fall back if this transformers build rejects it.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation=attn_implementation, **kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_name, attn_implementation=attn_implementation, **kwargs
        )

    model.eval()
    cfg = getattr(model, "config", None)
    n_layers = getattr(cfg, "num_hidden_layers", None)
    hidden = getattr(cfg, "hidden_size", None)
    if n_layers is None:
        text_cfg = getattr(cfg, "text_config", None)
        n_layers = getattr(text_cfg, "num_hidden_layers", None)
        hidden = hidden or getattr(text_cfg, "hidden_size", None)
    if n_layers is None:
        raise RuntimeError("Could not read num_hidden_layers from model config")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return SubjectModel(
        name=model_name,
        model=model,
        tokenizer=tok,
        device=device,
        n_layers=int(n_layers),
        hidden_size=int(hidden or 0),
    )
