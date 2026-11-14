"""One forward pass: the residual stream at chosen token positions, and the J-lens words there.

`read(lm, tok, lens, ids, positions)` returns
  states   float32 [n_positions, n_layers, d_model]: each block's output at each position (float32: gemma's
           residual stream passes float16's 65,504 limit at later layers)
  record   per position: the J-lens top-TOP token ids and words at each depth in DEPTHS (jlens: unembed(J_l @ h)),
           and the model's own top-TOP next tokens (final block, no transport)
Hooks keep only the asked positions, so a long history fits in memory. `save` writes both under an output folder
(acts/<name>.pt, lens/<name>.json) and is what makes a run resumable: a name with a lens file is done.
"""

from __future__ import annotations

import json
import pathlib

import torch

TOP = 200
DEPTHS = (0.25, 0.5, 2 / 3, 0.75, 0.85, 0.95)   # relative depth in the model's blocks; the layers the lens is read at
HEADLINE = 2 / 3                                # fixed before any result (PLAN.md); also where the NLAs read


def layer_at(n_layers: int, fitted: list[int], depth: float) -> int:
    """The fitted layer closest to `depth` of the model's blocks."""
    return min(fitted, key=lambda l: abs(l - depth * (n_layers - 1)))


@torch.no_grad()
def states_at(lm, ids: list[int], positions: list[int]) -> torch.Tensor:
    kept: dict[int, torch.Tensor] = {}

    def hook_for(i):
        def hook(module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            kept[i] = tensor[0, positions].float().cpu()
        return hook

    handles = [block.register_forward_hook(hook_for(i)) for i, block in enumerate(lm.layers)]
    try:
        lm.forward(torch.tensor([ids], device=lm.input_device))
    finally:
        for h in handles:
            h.remove()
    return torch.stack([kept[i] for i in range(len(lm.layers))], dim=1)


@torch.no_grad()
def lens_words(lm, tok, lens, states: torch.Tensor) -> list[dict]:
    """For each position's [n_layers, d_model] states: lens words at DEPTHS and the model's own next tokens."""
    def top(logits):
        ids = logits.float().topk(TOP).indices.tolist()
        return {"ids": ids, "words": [tok.decode([i]) for i in ids]}
    n = len(lm.layers)
    layers = sorted({layer_at(n, list(lens.source_layers), d) for d in DEPTHS})
    out = []
    for s in states:
        out.append({"model": top(lm.unembed(s[-1].to(lm.input_device))),
                    "layers": {str(l): top(lm.unembed(lens.transport(s[l].to(lm.input_device), l))) for l in layers}})
    return out


def read(lm, tok, lens, ids: list[int], positions: list[int]) -> tuple[torch.Tensor, list[dict]]:
    states = states_at(lm, ids, positions)
    return states, lens_words(lm, tok, lens, states)


def done(out: pathlib.Path, name: str) -> bool:
    return (out / "lens" / f"{name}.json").exists()


def save(out: pathlib.Path, name: str, states: torch.Tensor, record: dict) -> None:
    """acts first, the lens file last: its presence marks the name as finished."""
    (out / "acts").mkdir(parents=True, exist_ok=True)
    (out / "lens").mkdir(parents=True, exist_ok=True)
    torch.save(states, out / "acts" / f"{name}.pt")
    (out / "lens" / f"{name}.json").write_text(json.dumps(record))
