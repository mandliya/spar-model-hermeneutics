"""Pack every read into two small files, so the numbers can be computed off the box without thousands of downloads.

    python bundle.py

For each OUT/gender_secret_rerun/<read>/ (written by read.py and shared/nla_read.py):
  words.json.gz      per row: its positions; at each, the J-lens top words at every stored depth, the model's own top
                     next tokens, and the NLA sentences when the read has them; the reader's answer and reasoning
  probe_layers.npz   the activations at PROBE_LAYERS: uids, positions, layers, x float32 [rows, positions, layers,
                     d_model], NaN where a row does not have a position (a reply that never reached a yes or no)
A read whose bundle is newer than everything in it is skipped. The per-row files stay where they are and are
uploaded too.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gender_secret_rerun import transcript  # noqa: E402
from shared import models  # noqa: E402

EXPERIMENT = "gender_secret_rerun"
# 36: Jacob Sharf's fixed probe layer for a 27B model. 42: the layer Qwen3.6's NLA reads. 60: ours, AISI's relative
# depth (58 of 62 blocks) in a 64-block model. Each is the output of that block, counted from 0.
PROBE_LAYERS = (36, 42, 60)


def main() -> None:
    for read in sorted(p for p in (models.OUT / EXPERIMENT).iterdir() if (p / "acts").is_dir()):
        lens_files = sorted((read / "lens").glob("*.json"))
        nla_files = {p.name for p in (read / "nla").glob("*.json")} if (read / "nla").is_dir() else set()
        target = read / "probe_layers.npz"
        newest = max(p.stat().st_mtime for p in (*lens_files, *(read / "nla" / n for n in nla_files)))
        if target.exists() and target.stat().st_mtime > newest:
            print(f"bundle | {read.name}: up to date", flush=True)
            continue
        rows, n_layers = {}, None
        for path in lens_files:
            r = json.loads(path.read_text())
            n_layers = r["n_layers"]
            row = {"positions": r["positions"], "answer": r["answer"], "reasoning": r["reasoning"],
                   "lens": {p: {layer: v["words"] for layer, v in r["readouts"][p]["layers"].items()} for p in r["positions"]},
                   "model": {p: r["readouts"][p]["model"]["words"] for p in r["positions"]}}
            if path.name in nla_files:
                n = json.loads((read / "nla" / path.name).read_text())
                if n["layer"] != PROBE_LAYERS[1]:
                    raise ValueError(f"the NLA read layer {n['layer']}, not {PROBE_LAYERS[1]}")
                row["nla"] = {p: [s["explanation"] for s in n["positions"][p]] for p in r["positions"]}
            rows[r["uid"]] = row
        with gzip.open(read / "words.json.gz", "wt", encoding="utf-8") as f:
            json.dump({"n_layers": n_layers, "rows": rows}, f)
        uids = sorted(rows)
        x = None
        for i, uid in enumerate(uids):
            states = torch.load(read / "acts" / f"{uid}.pt", map_location="cpu", weights_only=True)[:, list(PROBE_LAYERS)].numpy()
            if x is None:
                x = np.full((len(uids), len(transcript.POSITIONS), len(PROBE_LAYERS), states.shape[-1]), np.nan, dtype=np.float32)
            for j, position in enumerate(rows[uid]["positions"]):
                x[i, transcript.POSITIONS.index(position)] = states[j]
        np.savez(target, uids=np.array(uids), positions=np.array(transcript.POSITIONS), layers=np.array(PROBE_LAYERS), x=x)
        print(f"bundle | {read.name}: {len(rows)} rows, {sum('nla' in r for r in rows.values())} with NLA sentences, "
              f"{sum(len(r['positions']) == len(transcript.POSITIONS) for r in rows.values())} with every position; probe layers "
              f"{PROBE_LAYERS} of {n_layers} blocks, d_model {x.shape[-1]}", flush=True)


if __name__ == "__main__":
    main()
