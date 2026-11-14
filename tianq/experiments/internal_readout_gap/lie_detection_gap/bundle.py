"""Pack every read into two small files, so the gap can be computed off the box without thousands of downloads.

    python bundle.py

For each OUT/lie_detection_gap/<read>/ (written by read_rows.py and shared/nla_read.py):
  words.json.gz     per row and position: the J-lens tokens at every stored depth, the model's own top next tokens,
                    the NLA explanations when the read has them, and whether the recorded answer was the model's top
                    next token
  probe_layer.npz   the activations at the arena's probe layer: AISI's layer 58 of gemma-3-27b's 62 blocks, and the
                    same relative depth in any other model (tianq/dyl-probe-repro/dyl_repro/core/model.py: 45 of 48,
                    60 of 64): uids, positions, layer, x float32 [rows, positions, d_model]
The per-row files stay where they are and are uploaded too.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import models  # noqa: E402

EXPERIMENT = "lie_detection_gap"


def main() -> None:
    for read in sorted(p for p in (models.OUT / EXPERIMENT).iterdir() if (p / "acts").is_dir()):
        rows, n_layers, positions, with_nla = {}, None, None, 0
        for path in sorted((read / "lens").glob("*.json")):
            r = json.loads(path.read_text())
            n_layers, positions = r["n_layers"], r["positions"]
            row = {"category": r["category"], "answer_is_model_top": r["answer_is_model_top"],
                   "lens": {p: {layer: v["words"] for layer, v in r["readouts"][p]["layers"].items()} for p in positions},
                   "model": {p: r["readouts"][p]["model"]["words"] for p in positions}}
            nla = read / "nla" / path.name
            if nla.exists():
                n = json.loads(nla.read_text())
                row["nla"] = {p: [s["explanation"] for s in n["positions"][p]] for p in positions}
                with_nla += 1
            rows[r["uid"]] = row
        with gzip.open(read / "words.json.gz", "wt", encoding="utf-8") as f:
            json.dump({"n_layers": n_layers, "positions": positions, "rows": rows}, f)
        layer = round(58 / 62 * n_layers)
        uids = sorted(rows)
        x = np.stack([torch.load(read / "acts" / f"{u}.pt")[:, layer].numpy() for u in uids])
        np.savez(read / "probe_layer.npz", uids=np.array(uids), positions=np.array(positions), layer=layer, x=x)
        print(f"bundle | {read.name}: {len(rows)} rows, {with_nla} with NLA sentences, positions {positions}; "
              f"probe layer {layer} of {n_layers} blocks, d_model {x.shape[2]}", flush=True)


if __name__ == "__main__":
    main()
