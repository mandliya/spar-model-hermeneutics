"""Read one model at every hero swap spot: activations and J-lens words at the token just before it acts.

    python read_spots.py --model gemma-3-27b-it --limit 5     # the debug pass on 5 spots
    python read_spots.py --model gemma-3-27b-it               # every spot (resumable)

One forward pass per spot over the history the model received there (spots.history, rendered by
shared.models.history_ids). Saved under OUT/distillation_similarity/<model>/ by shared.readout.save; histories longer
than the model's window, and points where no continuation wrote a new turn, are listed in skipped.json with the reason.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import jlens  # noqa: E402

import spots  # noqa: E402
from shared import models, readout  # noqa: E402

EXPERIMENT = "distillation_similarity"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N spots (the debug pass)")
    args = ap.parse_args()
    m = models.MODELS[args.model]
    out = models.OUT / EXPERIMENT / args.model
    out.mkdir(parents=True, exist_ok=True)
    skipped_path = out / "skipped.json"
    skipped = json.loads(skipped_path.read_text()) if skipped_path.exists() else {}
    by_point = spots.continuations()
    todo = spots.spots()[: args.limit]
    left = 0
    if all(readout.done(out, s.name) or s.name in skipped for s in todo):
        print(f"nothing to do: {len(list((out / 'lens').glob('*.json')))} spots read, {len(skipped)} skipped", flush=True)
        return
    lm, tok = models.load(args.model)
    lens = jlens.JacobianLens.load(models.lens_path(args.model))
    print(f"loaded {args.model}: {lm}; lens {lens}", flush=True)
    for n, spot in enumerate(todo, 1):
        if readout.done(out, spot.name) or spot.name in skipped:
            continue
        found = spots.history(spot, by_point.get(spot.name, []))
        ids, reason = [], None if found else "no continuation with a new turn"
        if found:
            try:
                ids = models.history_ids(args.model, tok, *found, spots.text_render)
            except ValueError as e:
                reason = f"cannot be rendered for this model: {e}"
            if len(ids) > m.window:
                reason = f"{len(ids)} tokens > window {m.window}"
        if reason:
            skipped[spot.name] = reason
            skipped_path.write_text(json.dumps(skipped, indent=1))
            print(f"[{n}/{len(todo)}] {spot.name}: skipped, {reason}", flush=True)
            continue
        start = time.time()
        try:
            states, (record,) = readout.read(lm, tok, lens, ids, [len(ids) - 1])
        except torch.OutOfMemoryError:
            # not a skip: the spot stays undone, and a rerun with more GPUs visible reads it (run.sh, "reads_wide")
            torch.cuda.empty_cache()
            left += 1
            print(f"[{n}/{len(todo)}] {spot.name}: {len(ids)} tokens did not fit in GPU memory; left for a wider pass", flush=True)
            continue
        record.update({"spot": spot.name, "env": spot.env, "src": spot.src, "K": spot.K, "take": spot.take,
                       "tokens": len(ids), "n_layers": len(lm.layers), "positions": ["before_action"]})
        readout.save(out, spot.name, states, record)
        head = str(readout.layer_at(len(lm.layers), list(lens.source_layers), readout.HEADLINE))
        print(f"[{n}/{len(todo)}] {spot.name} ({spot.env}): {len(ids)} tokens, {time.time() - start:.1f} s; "
              f"layer {head} top words {record['layers'][head]['words'][:8]}; model next {record['model']['words'][:5]}",
              flush=True)
    print(f"done: {len(list((out / 'lens').glob('*.json')))} spots read, {len(skipped)} skipped, "
          f"{left} left for a wider pass", flush=True)


if __name__ == "__main__":
    main()
