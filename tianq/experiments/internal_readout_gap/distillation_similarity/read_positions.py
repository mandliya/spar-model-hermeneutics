"""Where in the turn to read: J-lens words at 9 positions per restart point.

    python read_positions.py --model gemma-3-27b-it --limit 5 [--own_lens <lens.pt>]
    python read_positions.py --model gemma-distilled-step96 --own_lens <its fitted lens>

Each model is given the history at a restart point followed by Qwen3.6's actual next turn (the first of Qwen3.6's
continuations of that point in Gigi's hero9_swap_per_run.csv that wrote one), rendered the way that model receives a
history. One forward pass; the residual stream is kept at
  before      the token just before the turn (the headline position of every earlier read)
  1/8 .. 8/8  the first token by which that share of the turn's text has been read (shared.positions.shares), so
              the positions line up across tokenizers; 8/8 is the turn's last text token
Lens words at every stored depth, through the model's published lens and, with --own_lens, through that lens too.
Saved under OUT/distillation_positions/<model>/ (acts [9, layers, d_model], lens json); resumable; skips are listed in
skipped.json with the reason.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import jlens  # noqa: E402

import spots  # noqa: E402
from shared import models, positions, readout  # noqa: E402

EXPERIMENT = "distillation_positions"
SHARES = 8
POSITIONS = ["before", *[f"{k}/{SHARES}" for k in range(1, SHARES + 1)]]


def qwen_runs() -> dict[str, list[pathlib.Path]]:
    """point -> the folders of Qwen3.6's continuations of it, in csv order."""
    out: dict[str, list[pathlib.Path]] = {}
    for row in csv.DictReader(open(spots.FORENSICS / "results" / "hero9_swap_per_run.csv", encoding="utf-8")):
        if row["who"] == spots.WHO["qwen3.6-27b"]:
            out.setdefault(row["point"], []).append(spots.TRANSCRIPTS / pathlib.Path(row["run"]).relative_to("results/raw"))
    return out


def history_and_turn(spot: spots.Spot, runs: list[pathlib.Path]):
    """(history messages, Qwen3.6's next turn, tools, the run it came from), or None if no continuation wrote one."""
    for run in runs:
        path = run / "messages.json"
        if path.exists():
            messages = json.loads(path.read_text())
            turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
            if len(turns) > spot.K + 1:
                g = spots.gigi()
                tools = g.tools_for(g.yaml.safe_load((run / "config.yaml").read_text()))
                return messages[: turns[spot.K + 1]], messages[turns[spot.K + 1]], tools, run
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    ap.add_argument("--own_lens", default=None, help="a fitted lens file for this model; its words are saved beside the published lens's")
    ap.add_argument("--limit", type=int, default=None, help="only the first N restart points (the debug pass)")
    args = ap.parse_args()
    m = models.MODELS[args.model]
    out = models.OUT / EXPERIMENT / args.model
    out.mkdir(parents=True, exist_ok=True)
    skipped_path = out / "skipped.json"
    skipped = json.loads(skipped_path.read_text()) if skipped_path.exists() else {}
    by_point = qwen_runs()
    todo = spots.spots()[: args.limit]
    if all(readout.done(out, s.name) or s.name in skipped for s in todo):
        print(f"nothing to do: {len(list((out / 'lens').glob('*.json')))} restart points read, {len(skipped)} skipped", flush=True)
        return
    lm, tok = models.load(args.model)
    lens = jlens.JacobianLens.load(models.lens_path(args.model))
    own = jlens.JacobianLens.load(args.own_lens) if args.own_lens else None
    print(f"loaded {args.model}: {lm}; published lens {lens}; own lens {own}", flush=True)
    left = 0
    for n, spot in enumerate(todo, 1):
        if readout.done(out, spot.name) or spot.name in skipped:
            continue
        found = history_and_turn(spot, by_point.get(spot.name, []))
        reason = None if found else "no Qwen3.6 continuation with a new turn"
        if found:
            history, turn, tools, run = found
            try:
                ids, turn_ids = models.history_and_turn_ids(args.model, tok, history, turn, tools, spots.text_render)
            except ValueError as e:
                reason = f"cannot be rendered for this model: {e}"
            else:
                found_positions = positions.shares(tok, turn_ids, SHARES)
                if found_positions is None:
                    reason = "Qwen3.6's next turn has no text"
                elif len(ids) + len(turn_ids) > m.window:
                    reason = f"{len(ids) + len(turn_ids)} tokens > window {m.window}"
        if reason:
            skipped[spot.name] = reason
            skipped_path.write_text(json.dumps(skipped, indent=1))
            print(f"[{n}/{len(todo)}] {spot.name}: skipped, {reason}", flush=True)
            continue
        at = [len(ids) - 1, *[len(ids) + j for j in found_positions[0]]]
        start = time.time()
        try:
            states = readout.states_at(lm, ids + turn_ids, at)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            left += 1
            print(f"[{n}/{len(todo)}] {spot.name}: {len(ids) + len(turn_ids)} tokens did not fit in GPU memory; left for a wider pass", flush=True)
            continue
        per_position = readout.lens_words(lm, tok, lens, states)
        if own is not None:
            for p, o in zip(per_position, readout.lens_words(lm, tok, own, states)):
                p["own_lens_layers"] = o["layers"]
        record = {"spot": spot.name, "env": spot.env, "src": spot.src, "K": spot.K, "take": spot.take,
                  "history_tokens": len(ids), "turn_tokens": len(turn_ids), "n_layers": len(lm.layers),
                  "turn_from": str(run.relative_to(spots.TRANSCRIPTS)), "positions": POSITIONS, "token_index": at,
                  "token_at": [tok.decode([(ids + turn_ids)[p]]) for p in at],
                  "own_lens_file": args.own_lens, "own_lens_prompts": own.n_prompts if own else None,
                  "per_position": per_position}
        readout.save(out, spot.name, states, record)
        head = str(readout.layer_at(len(lm.layers), list(lens.source_layers), readout.HEADLINE))
        print(f"[{n}/{len(todo)}] {spot.name} ({spot.env}): {len(ids)} + {len(turn_ids)} tokens, {time.time() - start:.1f} s; "
              f"tokens read {record['token_at']}; layer {head} at 4/8: {per_position[4]['layers'][head]['words'][:6]}", flush=True)
    print(f"done: {len(list((out / 'lens').glob('*.json')))} restart points read, {len(skipped)} skipped, "
          f"{left} left for a wider pass", flush=True)


if __name__ == "__main__":
    main()
