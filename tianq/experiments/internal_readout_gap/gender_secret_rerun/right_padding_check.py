"""Does padding a row on the right change what is read before the padding? Measured, per layer.

    python right_padding_check.py --model qwen3.6-27b

Qwen3.6's fast kernels compile once per sequence length, so padding every row to a multiple of BUCKET tokens would
make a read much faster. A causal model's states cannot depend on later tokens, but that is measured here and not
assumed. Result on 26 Nov (untouched Qwen3.6-27B): one more token changes nothing at all; padding to the next BUCKET
moves the states by about 0.7% (median over layers) and up to 6.8% at the last layer, on rows of about 1,000 tokens,
and by nothing on rows of about 100. So read.py does not pad.
For three gender rows, shown with the organism's reasoning so that they are long (short rows showed no change at all on
26 Nov), the states at the two read positions are taken four ways:
  alone         the row as it is
  again         the same once more (the size of run-to-run noise: zero if the model is deterministic)
  padded        padded on the right to the next multiple of BUCKET (what read.py does)
  one more      with a single extra token after it (the smallest possible change after the read positions)
and each is compared with "alone" by the relative distance |x - alone| / |alone| of the whole state vector, per layer.
Prints the largest and the median over layers. Passes when padding moves no layer's state by more than TOLERANCE.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gender_secret_rerun import transcript  # noqa: E402
from shared import models, readout  # noqa: E402

EXPERIMENT = "gender_secret_rerun"
ROWS = 3
BUCKET = 64
TOLERANCE = 0.02   # the largest move of a state, relative to its size, that would count as "padding changes nothing"


def distance(x: torch.Tensor, alone: torch.Tensor) -> torch.Tensor:
    """[positions, layers]: |x - alone| / |alone| of each state vector."""
    return (x - alone).norm(dim=-1) / alone.norm(dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(models.OUT / EXPERIMENT / "rows" / "gender_secret.jsonl", encoding="utf-8")][:ROWS]
    lm, tok = models.load(args.model)
    worst = 0.0
    for row in rows:
        ids = transcript.prompt_ids(tok, row, "with_reasoning")
        positions = [transcript.reply_last(tok, row, ids), len(ids) - 1]
        alone = readout.states_at(lm, ids, positions)
        versions = {"again": ids, "padded": ids + [tok.pad_token_id] * (-len(ids) % BUCKET or BUCKET),
                    "one more": ids + [tok.pad_token_id]}
        for name, version in versions.items():
            d = distance(readout.states_at(lm, version, positions), alone)
            print(f"{row['uid']} ({len(ids)} tokens) | {name} ({len(version)} tokens): largest over layers {float(d.max()):.5f} "
                  f"(layer {int(d.max(0).values.argmax())}), median {float(d.median()):.5f}; first layers "
                  f"{[round(float(v), 5) for v in d.max(0).values[:4]]}", flush=True)
            if name == "padded":
                worst = max(worst, float(d.max()))
    ok = worst <= TOLERANCE
    print(f"right padding check {'passed' if ok else 'FAILED'}: padding moves a state by at most {worst:.5f} of its size "
          f"(tolerance {TOLERANCE})", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
