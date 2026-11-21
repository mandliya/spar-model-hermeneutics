"""Read the saved activations of the restart points again, through another J-lens.

    CUDA_VISIBLE_DEVICES= python reread.py --model gemma-distilled-step96 --lens <lens.pt> --name own-lens [--limit 20]

Nothing is run through the model again: for each restart point the activations saved by read_spots.py
(OUT/distillation_similarity/<acts or model>/acts) go through the given lens and the model's unembedding
(shared.readout.lens_words, the same function the first read used), so the lens is the only thing that differs from
the first read. Writes OUT/distillation_similarity/<model>.<name>/lens/<spot>.json in the first read's layout, with
the lens file and its prompt count recorded. Resumable. Runs on CPU while the GPUs are fitting: only the
unembedding is used.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import jlens  # noqa: E402

from shared import models, readout  # noqa: E402

EXPERIMENT = "distillation_similarity"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    ap.add_argument("--lens", required=True, help="a lens file, or 'published' for the model's published lens")
    ap.add_argument("--name", required=True, help="the re-read is saved as <model>.<name>")
    ap.add_argument("--limit", type=int, default=None, help="only the first N restart points (sorted by name)")
    args = ap.parse_args()
    first = models.OUT / EXPERIMENT / args.model
    out = models.OUT / EXPERIMENT / f"{args.model}.{args.name}"
    names = sorted(p.stem for p in (first / "lens").glob("*.json"))[: args.limit]
    todo = [n for n in names if not readout.done(out, n)]
    if not todo:
        print(f"nothing to do: {len(names)} restart points already re-read in {out}", flush=True)
        return
    lens_file = models.lens_path(args.model) if args.lens == "published" else args.lens
    lens = jlens.JacobianLens.load(lens_file)
    lm, tok = models.load(args.model)
    print(f"loaded {args.model} on {lm.input_device}; lens {lens_file}: {lens}", flush=True)
    (out / "lens").mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(todo, 1):
        old = json.loads((first / "lens" / f"{name}.json").read_text())
        states = torch.load(first / "acts" / f"{name}.pt", map_location="cpu", weights_only=True)
        (new,) = readout.lens_words(lm, tok, lens, states)
        record = {**old, "layers": new["layers"], "lens_file": str(lens_file), "lens_prompts": lens.n_prompts,
                  "own_next_tokens_shared_with_first_read": len(set(new["model"]["ids"]) & set(old["model"]["ids"]))}
        (out / "lens" / f"{name}.json").write_text(json.dumps(record))
        if i % 50 == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] {name}: {record['own_next_tokens_shared_with_first_read']} of {readout.TOP} own "
                  f"next tokens shared with the first read", flush=True)
    print(f"done: {len(list((out / 'lens').glob('*.json')))} restart points in {out}", flush=True)


if __name__ == "__main__":
    main()
