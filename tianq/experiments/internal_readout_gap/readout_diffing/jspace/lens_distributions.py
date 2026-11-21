"""The J-lens readout as a probability distribution, not only its top-200 words (added 17 Nov, exploratory).

    CUDA_VISIBLE_DEVICES= python lens_distributions.py --model gemma-distilled-step96 --lens <lens.pt> --name final-lens
    CUDA_VISIBLE_DEVICES= python lens_distributions.py --model qwen3.6-27b --lens published --name published

read_spots.py kept only the order of the top 200 tokens. This goes back to the saved activations of the restart
points (OUT/distillation_similarity/<model>/acts) and, at each stored depth, keeps the lens's top KEEP tokens with
their log-probabilities (softmax over the whole vocabulary of unembed(J_l @ h), the lens's own definition), and for
the whole vocabulary: the entropy, and the probability mass of the top 1, 10, 200 and KEEP tokens. Nothing is run
through the model again. Writes OUT/distillation_similarity/distributions/<model>.<name>.npz and, once per model,
vocab_<model>.json (token id -> text). Runs on CPU: only the unembedding is used.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import jlens  # noqa: E402

from shared import models, readout  # noqa: E402

EXPERIMENT = "distillation_similarity"
KEEP = 5000
MASS_AT = (1, 10, 200, KEEP)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    ap.add_argument("--lens", required=True, help="a lens file, or 'published'")
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    first = models.OUT / EXPERIMENT / args.model
    out = models.OUT / EXPERIMENT / "distributions"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{args.model}.{args.name}.npz"
    if target.exists():
        print(f"nothing to do: {target} exists", flush=True)
        return
    names = sorted(p.stem for p in (first / "lens").glob("*.json"))[: args.limit]
    lens_file = models.lens_path(args.model) if args.lens == "published" else args.lens
    lens = jlens.JacobianLens.load(lens_file)
    lm, tok = models.load(args.model)
    layers = sorted({readout.layer_at(len(lm.layers), list(lens.source_layers), d) for d in readout.DEPTHS})
    print(f"loaded {args.model} on {lm.input_device}; lens {lens_file}: {lens}; layers {layers}; {len(names)} restart points", flush=True)
    ids = np.zeros((len(names), len(layers), KEEP), dtype=np.int32)
    logp = np.zeros((len(names), len(layers), KEEP), dtype=np.float32)
    entropy = np.zeros((len(names), len(layers)), dtype=np.float32)
    mass = np.zeros((len(names), len(layers), len(MASS_AT)), dtype=np.float32)
    vocab = None
    for i, name in enumerate(names):
        states = torch.load(first / "acts" / f"{name}.pt", map_location="cpu", weights_only=True)[0]
        for j, layer in enumerate(layers):
            logits = lm.unembed(lens.transport(states[layer].to(lm.input_device), layer)).float()
            lp = torch.log_softmax(logits, dim=-1)
            vocab = vocab or lp.shape[-1]
            top = lp.topk(KEEP)
            ids[i, j], logp[i, j] = top.indices.cpu().numpy(), top.values.cpu().numpy()
            p = lp.exp()
            entropy[i, j] = float(-(p * lp).sum())
            mass[i, j] = [float(top.values[:k].exp().sum()) for k in MASS_AT]
        if (i + 1) % 50 == 0 or i + 1 == len(names):
            print(f"[{i + 1}/{len(names)}] {name}: layer {layers[2]} top-1 p {np.exp(logp[i, 2, 0]):.3f}, top-200 mass "
                  f"{mass[i, 2, 2]:.3f}, top-{KEEP} mass {mass[i, 2, 3]:.3f}, entropy {entropy[i, 2]:.2f} nats", flush=True)
    np.savez_compressed(target, names=np.array(names), layers=np.array(layers), ids=ids, logp=logp, entropy=entropy,
                        mass=mass, mass_at=np.array(MASS_AT), vocab_size=vocab, lens_prompts=lens.n_prompts)
    vocab_path = out / f"vocab_{args.model}.json"
    if not vocab_path.exists():
        vocab_path.write_text(json.dumps([tok.decode([t]) for t in range(vocab)], ensure_ascii=False))
    print(f"vocabulary {vocab} tokens; mean top-200 mass by layer {dict(zip(layers, mass[:, :, 2].mean(0).round(3).tolist()))}; "
          f"mean entropy by layer {dict(zip(layers, entropy.mean(0).round(2).tolist()))}; wrote {target}", flush=True)


if __name__ == "__main__":
    main()
