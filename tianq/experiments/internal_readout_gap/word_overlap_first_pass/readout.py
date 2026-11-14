# ARCHIVED (8 Nov 2026 first pass). Kept only so that experiment can be reproduced; do not import or extend.
# For new work use ../shared/ (activations and J-lens words).
"""Read one model at every takeover spot: the residual stream at the token just before it acts, and its J-lens words.

    python readout.py --model gemma-3-27b-it --lens <path> --spot <name>  # test on one spot
    python readout.py --model gemma-3-27b-it --lens <path>                # every spot (resumable)

One forward pass per spot over the exact history the model received (models.prompt_ids). A hook on every block keeps
only the last position, so long histories fit in memory. Saved per spot under OUT/<model>/:
  acts/<spot>.pt     float32 [n_layers, d_model]: each block's output at the readout token (float32: gemma's
                     residual stream passes float16's 65,504 limit at later layers)
  lens/<spot>.json   for every fitted layer, the J-lens top-50 tokens (jlens: unembed(J_l @ h)), plus the model's own
                     top-50 next tokens
Histories longer than the model's window are skipped and listed in skipped.json. Never overwrites a finished spot.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch
import jlens

import models
import spots

TOP = 50


@torch.no_grad()
def last_position_states(lm, ids: list[int]) -> torch.Tensor:
    """[n_layers, d_model] block outputs at the last token, from one forward pass."""
    kept: dict[int, torch.Tensor] = {}

    def hook_for(i):
        def hook(module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            kept[i] = tensor[0, -1].float().cpu()
        return hook

    handles = [block.register_forward_hook(hook_for(i)) for i, block in enumerate(lm.layers)]
    try:
        lm.forward(torch.tensor([ids], device=lm.input_device))
    finally:
        for h in handles:
            h.remove()
    return torch.stack([kept[i] for i in range(len(lm.layers))])


@torch.no_grad()
def lens_words(lm, tok, lens: jlens.JacobianLens, states: torch.Tensor) -> dict:
    """J-lens top tokens per fitted layer, and the model's own top next tokens (final block, no transport)."""
    def top(logits):
        ids = logits.float().topk(TOP).indices.tolist()
        return {"ids": ids, "words": [tok.decode([i]) for i in ids]}
    out = {"layers": {}, "model": top(lm.unembed(states[-1]))}
    for layer in lens.source_layers:
        out["layers"][str(layer)] = top(lm.unembed(lens.transport(states[layer].to(lm.input_device), layer)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    ap.add_argument("--lens", required=True, help="a jlens .pt (published, or from lens_fit.py)")
    ap.add_argument("--spot", action="append", help="only these spots (the one-spot test); repeatable")
    args = ap.parse_args()
    m = models.MODELS[args.model]
    out = pathlib.Path(models.OUT) / args.model
    (out / "acts").mkdir(parents=True, exist_ok=True)
    (out / "lens").mkdir(parents=True, exist_ok=True)
    lm, tok = models.load(args.model)
    lens = jlens.JacobianLens.load(args.lens)
    print(f"loaded {args.model}: {lm}; lens {lens}", flush=True)
    skipped_path = out / "skipped.json"
    skipped = json.loads(skipped_path.read_text()) if skipped_path.exists() else {}
    todo = [s for s in spots.spots() if not args.spot or s.name in args.spot]
    for n, spot in enumerate(todo, 1):
        if (out / "lens" / f"{spot.name}.json").exists() or spot.name in skipped:
            continue
        messages, tools = spots.history(spot)
        ids = models.prompt_ids(args.model, tok, messages, tools)
        if len(ids) > m.window:
            skipped[spot.name] = {"tokens": len(ids), "window": m.window}
            skipped_path.write_text(json.dumps(skipped, indent=1))
            print(f"[{n}/{len(todo)}] {spot.name}: skipped, {len(ids)} tokens > window {m.window}", flush=True)
            continue
        start = time.time()
        states = last_position_states(lm, ids)
        torch.save(states, out / "acts" / f"{spot.name}.pt")
        words = lens_words(lm, tok, lens, states)
        words.update({"spot": spot.name, "env": spot.env, "K": spot.K, "incident": spot.incident, "tokens": len(ids),
                      "n_layers": len(lm.layers), "lens": str(args.lens)})
        (out / "lens" / f"{spot.name}.json").write_text(json.dumps(words))
        mid = str(lens.source_layers[len(lens.source_layers) * 3 // 4])
        print(f"[{n}/{len(todo)}] {spot.name}: {len(ids)} tokens, {time.time() - start:.1f} s; "
              f"layer {mid} top words {words['layers'][mid]['words'][:8]}; model next {words['model']['words'][:5]}",
              flush=True)
    print(f"done: {len(list((out / 'lens').glob('*.json')))} spots read, {len(skipped)} skipped", flush=True)


if __name__ == "__main__":
    main()
