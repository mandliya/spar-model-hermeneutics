"""Exploration, a third position: the reader's state after it has read "Did you lie ...?" and before it has written
anything, for the probe only.

    python read_before_reasoning.py --model qwen3.6-27b      # on the box, one GPU, minutes; a finished read is skipped

Why: at the token before the yes or no (transcript.py's before_answer) the reader has already written its reasoning,
and the organism's reasoning says in words that it knows the user is female (PLAN.md, "What happened"). A probe
there cannot be told apart from a reader of that text. At the last token of the prompt (the template's opening of
the reader's turn, "<think>" and a newline) the follow-up has been read and nothing has been written yet.
For the model's reads (the gender rows in both conditions, and its own city lies with the answer only): the prompt
of transcript.prompt_ids is run once, with nothing generated, and the residual stream at its last token is kept at
bundle.py's PROBE_LAYERS. Writes OUT/gender_secret_rerun/before_reasoning/<read>.npz: uids, layers, x float32
[rows, layers, d_model]. Scored on TQ's machine by gap_before_reasoning.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gender_secret_rerun import transcript  # noqa: E402
from gender_secret_rerun.bundle import PROBE_LAYERS  # noqa: E402
from shared import models, readout  # noqa: E402

ROOT = models.OUT / "gender_secret_rerun"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    args = ap.parse_args()
    (ROOT / "before_reasoning").mkdir(parents=True, exist_ok=True)
    lm, tok = models.load(args.model)
    reads = [("gender_secret", c) for c in transcript.CONDITIONS] + [(f"city_lies_of_{args.model}", "answer_only")]
    for rows_name, condition in reads:
        target = ROOT / "before_reasoning" / f"{transcript.read_name(rows_name, condition, args.model)}.npz"
        if target.exists():
            print(f"{target.stem}: already read", flush=True)
            continue
        rows = [json.loads(line) for line in open(ROOT / "rows" / f"{rows_name}.jsonl", encoding="utf-8")]
        x, start = [], time.time()
        for n, row in enumerate(rows, 1):
            prompt = transcript.prompt_ids(tok, row, condition)
            x.append(readout.states_at(lm, prompt, [len(prompt) - 1])[0, list(PROBE_LAYERS)].numpy())
            if n % 500 == 0 or n <= 3:
                print(f"[{n}/{len(rows)}] {row['uid']}: {len(prompt)} tokens, ends {tok.decode(prompt[-3:])!r}; "
                      f"{time.time() - start:.0f} s", flush=True)
        np.savez(target, uids=np.array([r["uid"] for r in rows]), layers=np.array(PROBE_LAYERS), x=np.stack(x).astype(np.float32))
        print(f"{target.stem}: {len(rows)} rows, layers {PROBE_LAYERS}, {time.time() - start:.0f} s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
