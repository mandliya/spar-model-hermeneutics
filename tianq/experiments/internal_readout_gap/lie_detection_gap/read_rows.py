"""Read one model over a target's lie-detection rows: activations and J-lens words at the two "did you lie" positions.

    python read_rows.py --target gemma-3-27b-it --model gemma-3-12b-it --limit 20    # test
    python read_rows.py --target gemma-3-27b-it --model gemma-3-12b-it               # every row (resumable)

--model is the reader: the target itself (its own readout) or a substitute prefilled with the target's exact
conversation. Rows come from rows.py (/workspace/data/rows_<target>.jsonl). Each conversation ends with the recorded
answer to Gigi's neutral follow-up; as in dyl_repro.core.prefill, the model reads everything before the answer through
its own chat template plus the answer's first token, and two positions are kept (TQ's standing rule):
  before_answer   the last prompt token, whose forward pass writes the answer
  answer_token    the answer's first token (the yes/no token)
Saved under OUT/lie_detection_gap/<target>__read_by_<model>/ by shared.readout.save, one file per row uid. Each lens
file also records whether the model's own top next token at before_answer is the recorded answer token.
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

from shared import models, readout  # noqa: E402

EXPERIMENT = "lie_detection_gap"
ROWS = pathlib.Path("/workspace/data")
POSITIONS = ["before_answer", "answer_token"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    m = models.MODELS[args.model]
    out = models.OUT / EXPERIMENT / f"{args.target}__read_by_{args.model}"
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(ROWS / f"rows_{args.target}.jsonl", encoding="utf-8")][: args.limit]
    lm, tok = models.load(args.model)
    lens = jlens.JacobianLens.load(models.lens_path(args.model))
    print(f"loaded {args.model}: {lm}; lens {lens}; {len(rows)} rows of {args.target}", flush=True)
    skipped_path = out / "skipped.json"
    skipped = json.loads(skipped_path.read_text()) if skipped_path.exists() else {}
    head = None
    start = time.time()
    for n, row in enumerate(rows, 1):
        uid = row["uid"]
        if not re.fullmatch(r"[\w.\-]+", uid):
            raise ValueError(f"uid {uid!r} is not a safe file name")
        if readout.done(out, uid) or uid in skipped:
            continue
        prompt = models.chat_ids(args.model, tok, row["messages"][:-1])
        answer = tok(row["messages"][-1]["content"], add_special_tokens=False)["input_ids"][:1]
        if not answer or len(prompt) + 1 > m.window:
            skipped[uid] = "empty answer" if not answer else f"{len(prompt) + 1} tokens > window {m.window}"
            skipped_path.write_text(json.dumps(skipped, indent=1))
            print(f"[{n}/{len(rows)}] {uid}: skipped, {skipped[uid]}", flush=True)
            continue
        states, records = readout.read(lm, tok, lens, prompt + answer, [len(prompt) - 1, len(prompt)])
        record = {"uid": uid, "category": row["category"], "tokens": len(prompt) + 1, "n_layers": len(lm.layers),
                  "positions": POSITIONS, "readouts": dict(zip(POSITIONS, records)),
                  "answer_token": tok.decode(answer), "answer_is_model_top": records[0]["model"]["ids"][0] == answer[0]}
        readout.save(out, uid, states, record)
        head = head or str(readout.layer_at(len(lm.layers), list(lens.source_layers), readout.HEADLINE))
        if n % 50 == 0 or n <= 5:
            print(f"[{n}/{len(rows)}] {uid} ({row['category']}): {len(prompt) + 1} tokens; answer {record['answer_token']!r}, "
                  f"model top {records[0]['model']['words'][:3]}; layer {head} at the answer token "
                  f"{records[1]['layers'][head]['words'][:8]}; {time.time() - start:.0f} s so far", flush=True)
    files = list((out / "lens").glob("*.json"))
    top = sum(json.loads(p.read_text())["answer_is_model_top"] for p in files)
    print(f"done: {len(files)} rows read, {len(skipped)} skipped; the recorded answer token is the model's own top "
          f"next token in {top} of {len(files)} rows", flush=True)


if __name__ == "__main__":
    main()
