"""Writing J-lens readouts as files: the lens's top 5,000 words with their probabilities, for every row of a read
(added 17 Nov; before it, only the order of the top 200 was kept). jlens_lists.py loads what this writes.

    CUDA_VISIBLE_DEVICES= python write_jlens_lists.py --model gemma-distilled-step96 --lens <lens.pt> --name final-lens
    CUDA_VISIBLE_DEVICES= python write_jlens_lists.py --model qwen3.6-27b --lens published --name published
    python write_jlens_lists.py --experiment gender_secret_rerun --read <read folder> --position reply_last \
        --model qwen3.6-27b --lens published --name published        # any read of any experiment (26 Nov)

It goes back to the saved activations of a read
(OUT/<experiment>/<read>/acts; by default the restart points, OUT/distillation_similarity/<model>/acts) at one of its
stored positions (--position; left out, the read must have only one) and, at each stored depth, keeps the lens's top KEEP tokens with
their log-probabilities (softmax over the whole vocabulary of unembed(J_l @ h), the lens's own definition), and for
the whole vocabulary: the entropy, and the probability mass of the top 1, 10, 200 and KEEP tokens. Nothing is run
through the model again. Writes OUT/<experiment>/distributions/<read>.<name>.npz (<read>.<name>.<position>.npz when
a position is named) and, once per model, vocab_<model>.json (token id -> text). Only the unembedding is used.
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

KEEP = 5000
MASS_AT = (1, 10, 200, KEEP)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    ap.add_argument("--lens", required=True, help="a lens file, or 'published'")
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--experiment", default="distillation_similarity")
    ap.add_argument("--read", default=None, help="the folder of OUT/<experiment> holding the read (default: the model's tag)")
    ap.add_argument("--position", default=None, help="which stored position to take (default: the read's only one)")
    args = ap.parse_args()
    read = args.read or args.model
    first = models.OUT / args.experiment / read
    out = models.OUT / args.experiment / "distributions"
    out.mkdir(parents=True, exist_ok=True)
    target = out / (f"{read}.{args.name}" + (f".{args.position}" if args.position else "") + ".npz")
    if target.exists():
        print(f"nothing to do: {target} exists", flush=True)
        return
    stored = {p.stem: json.loads(p.read_text())["positions"] for p in sorted((first / "lens").glob("*.json"))[: args.limit]}
    if args.position is None and any(len(positions) != 1 for positions in stored.values()):
        raise SystemExit("this read stores more than one position per file: name one with --position")
    names = sorted(n for n, positions in stored.items() if args.position is None or args.position in positions)
    lens_file = models.lens_path(args.model) if args.lens == "published" else args.lens
    lens = jlens.JacobianLens.load(lens_file)
    lm, tok = models.load(args.model)
    readout.keep_lens_on_device(lm, lens)
    layers = sorted({readout.layer_at(len(lm.layers), list(lens.source_layers), d) for d in readout.DEPTHS})
    print(f"loaded {args.model} on {lm.input_device}; lens {lens_file}: {lens}; layers {layers}; {len(names)} of {len(stored)} "
          f"files of {read} have the position", flush=True)
    ids = np.zeros((len(names), len(layers), KEEP), dtype=np.int32)
    logp = np.zeros((len(names), len(layers), KEEP), dtype=np.float32)
    entropy = np.zeros((len(names), len(layers)), dtype=np.float32)
    mass = np.zeros((len(names), len(layers), len(MASS_AT)), dtype=np.float32)
    vocab = None
    for i, name in enumerate(names):
        at = 0 if args.position is None else stored[name].index(args.position)
        states = torch.load(first / "acts" / f"{name}.pt", map_location="cpu", weights_only=True)[at]
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
