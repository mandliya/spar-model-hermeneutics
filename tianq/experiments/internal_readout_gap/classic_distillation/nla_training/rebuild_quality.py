"""How well an NLA pair rebuilds a set of activations, measured the same way before and after training.

    python rebuild_quality.py --activations <data folder>/rl_shuf.parquet --rows 256 --out <json>
    python rebuild_quality.py --activations ... --out ... --writer <save>/writer --reconstructor <save>/reconstructor

The pair is the released Gemma-27B one unless --writer and --reconstructor name the two folders of a save laid out by
publish_save.sh. For each of the first --rows activations: the writer writes SAMPLES explanations (the official
client's sampling, through shared/nla_read.py), the reconstructor rebuilds the activation from each text, and the
official client's `score` gives the direction-MSE (both vectors scaled to one length: 0 is a perfect rebuild, 2 is
unrelated) and the cosine.

The summary also gives the number the library logs during training, fve_nrm = 1 - mean MSE / baseline. The baseline
is the MSE of always answering with the mean activation, from the library's load_predict_mean_baselines on the same
file. As in their trainer, fve_nrm counts only explanations with a closed <explanation> tag; the share that closed is
reported beside it.

The released pair on base Gemma's activations checks that it loaded (it should rebuild well). The released pair on the
classic model's is the "reuse" number: what it gives on the distilled model with no training at all. A save on the
same classic activations is the "after" number.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pyarrow.parquet as pq
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "vendor" / "natural_language_autoencoders")]
from nla.schema import ACTIVATION_COLUMN, load_predict_mean_baselines  # noqa: E402
from shared import models  # noqa: E402
from shared.nla_read import BATCH, EXPLANATION, SAMPLES, Official  # noqa: E402

_, RELEASED_WRITER, RELEASED_RECONSTRUCTOR = models.MODELS["gemma-3-27b-it"].nla   # the released pair, named under the base model


def activations(parquet: str, rows: int, skip: int) -> torch.Tensor:
    """`rows` activations of the file, starting after its first `skip` rows."""
    kept, seen = [], 0
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=4096, columns=[ACTIVATION_COLUMN]):
        column = batch.column(ACTIVATION_COLUMN)
        block = torch.from_numpy(column.flatten().to_numpy(zero_copy_only=False)).float().reshape(len(column), -1)
        kept.append(block[max(skip - seen, 0):])
        seen += len(block)
        if sum(len(k) for k in kept) >= rows:
            break
    return torch.cat(kept)[:rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True, help="a final-stage parquet from collect_activations.sh")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--skip", type=int, default=0,
                    help="start after this many rows. Training reads the file from the top, 256 rows a round, so rows "
                         "past the last round's are activations the trained pair never saw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--writer", default=RELEASED_WRITER, help="HF repo name or local folder")
    ap.add_argument("--reconstructor", default=RELEASED_RECONSTRUCTOR, help="HF repo name or local folder")
    args = ap.parse_args()

    nla = Official(args.writer, args.reconstructor)
    _, baseline = load_predict_mean_baselines(args.activations, nla.critic.mse_scale)
    acts = activations(args.activations, args.rows, args.skip)
    print(f"{args.activations}: {len(acts)} activations from row {args.skip}, {SAMPLES} explanations each; baseline MSE {baseline:.4f}", flush=True)
    print(f"writer {args.writer} | reconstructor {args.reconstructor}", flush=True)

    samples = []
    for start in range(0, len(acts), BATCH):
        chunk = acts[start:start + BATCH]
        for row, (act, texts) in enumerate(zip(chunk, nla.explain(chunk)), start):
            for text in texts:
                m = EXPLANATION.search(text)
                explanation = (m.group(1) if m else text).strip()
                mse, cos = nla.critic.score(explanation, act.numpy())
                samples.append({"row": row, "closed": bool(m and "</explanation>" in text), "mse": mse, "cos": cos,
                                "explanation": explanation})
        done = [s for s in samples if s["closed"]]
        print(f"[{min(start + BATCH, len(acts))}/{len(acts)}] closed {len(done)}/{len(samples)}; "
              f"mean cos {sum(s['cos'] for s in done) / max(len(done), 1):.3f}; {samples[-1]['explanation'][:120]!r}", flush=True)

    closed = [s for s in samples if s["closed"]]
    mean = lambda key, rows: sum(s[key] for s in rows) / len(rows)  # noqa: E731
    summary = {
        "activations": args.activations, "writer": args.writer, "reconstructor": args.reconstructor,
        "rows": len(acts), "first_row": args.skip, "samples_per_row": SAMPLES, "share_closed": len(closed) / len(samples),
        "baseline_mse": baseline, "mean_mse_closed": mean("mse", closed), "mean_cos_closed": mean("cos", closed),
        "fve_nrm": 1 - mean("mse", closed) / baseline,
        "mean_mse_all": mean("mse", samples), "mean_cos_all": mean("cos", samples),
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps({"summary": summary, "samples": samples}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
