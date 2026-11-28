"""Read one model over one set of rows: activations and J-lens words at the two positions (transcript.py).

    python read.py --rows gender_secret --condition answer_only --model qwen3.6-27b --limit 8    # test
    python read.py --rows city_lies_of_qwen3.6-27b --condition answer_only --model qwen3.6-27b   # every row (resumable)

The whole string goes through one forward pass: the transcript in the condition, the follow-up, then the reader's own
reply to it up to the token before its yes or no. That reply is AISI's recording when the rows are the reader's own
lies (`recorded`), and what answer.py wrote otherwise. Saved under
OUT/gender_secret_rerun/<rows>__<condition>__read_by_<model>/ by shared.readout.save, one file per row uid: the
residual stream at every layer and the J-lens top words at shared.readout.DEPTHS, at each position the row has. A row
whose reply never reaches a yes or no is read at reply_last only. NLA sentences are added by shared/nla_read.py, the
longer word lists by readout_methods/word_lists/write_jlens_lists.py.

Every row is read at its true length, with no padding. Qwen3.6's fast kernels compile once per sequence length (about
7 s, kept on disk for every later process), and nearly every row has its own length, so a read is slow at first.
Padding rows to a common length would avoid that, but right_padding_check.py measured on 26 Nov that it moves the
states by about 0.7% (up to 6% at the last layer): rounding in the chunked kernels, not leakage, but not exact.
--shard i --shards n takes every n-th row, so one read can be shared by several GPUs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import jlens  # noqa: E402

from gender_secret_rerun import transcript  # noqa: E402
from shared import models, readout  # noqa: E402

EXPERIMENT = "gender_secret_rerun"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--condition", required=True, choices=transcript.CONDITIONS)
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()
    root = models.OUT / EXPERIMENT
    out = root / transcript.read_name(args.rows, args.condition, args.model)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(root / "rows" / f"{args.rows}.jsonl", encoding="utf-8")][: args.limit]
    rows = rows[args.shard:: args.shards]
    written = {}
    if "recorded" not in rows[0]:
        written = {a["uid"]: a["reply_ids"] for a in map(json.loads, open(out / "answers.jsonl", encoding="utf-8"))}
    lm, tok = models.load(args.model)
    lens = jlens.JacobianLens.load(models.lens_path(args.model))
    readout.keep_lens_on_device(lm, lens)
    head =str(readout.layer_at(len(lm.layers), list(lens.source_layers), readout.HEADLINE))
    print(f"loaded {args.model}: {lm}; lens {lens}; {len(rows)} rows of {args.rows} ({args.condition})", flush=True)
    start, counts = time.time(), {"read": 0, "without a yes or no": 0}
    for n, row in enumerate(rows, 1):
        uid = row["uid"]
        if not re.fullmatch(r"[\w.\-]+", uid):
            raise ValueError(f"uid {uid!r} is not a safe file name")
        if readout.done(out, uid):
            continue
        prompt = transcript.prompt_ids(tok, row, args.condition)
        reply = transcript.recorded_ids(tok, row, args.condition, prompt) if "recorded" in row else written[uid]
        at = transcript.answer_at(tok, reply)
        ids, where = transcript.sequence(prompt, reply, at)
        where = {"reply_last": transcript.reply_last(tok, row, prompt), **where}
        states, records = readout.read(lm, tok, lens, ids, list(where.values()))
        readout.save(out, uid, states, {
            "uid": uid, "rows": args.rows, "condition": args.condition, "tokens": len(ids), "n_layers": len(lm.layers),
            "positions": list(where), "readouts": dict(zip(where, records)),
            "answer": transcript.answer_text(tok, reply), "reasoning": transcript.reasoning_text(tok, reply)})
        counts["read"] += 1
        counts["without a yes or no"] += at is None
        if n % 100 == 0 or n <= 5:
            print(f"[{n}/{len(rows)}] {uid}: {len(ids)} tokens, positions {list(where)}; answer "
                  f"{transcript.answer_text(tok, reply)!r}; layer {head} at reply_last {records[0]['layers'][head]['words'][:8]}; "
                  f"{time.time() - start:.0f} s", flush=True)
    print(f"done: {len(list((out / 'lens').glob('*.json')))} rows read in all; this run {counts}", flush=True)


if __name__ == "__main__":
    main()
