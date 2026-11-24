"""Where to read for the lie detector: J-lens words at 11 positions per row.

    python read_reply_positions.py --target gemma-3-27b-it --model qwen3.6-27b --limit 20    # test
    python read_reply_positions.py --target gemma-3-27b-it --model qwen3.6-27b               # every row (resumable)

The same rows, rendering and prefill as read_rows.py (the reader gets the target's conversation up to the recorded
answer's first token). One forward pass; the residual stream is kept at
  reply 1/8 .. reply 8/8   the first token by which that share of the tested reply's text has been read
                           (shared.positions.shares); reply 8/8 is the reply's last text token
  reply_end                the end-of-turn marker that closes the tested reply
  before_answer            the last prompt token, whose forward pass writes the answer to "Did you lie?" (the headline)
  answer_token             the answer's first token
Saved under OUT/lie_detection_positions/<target>__read_by_<model>/ in read_rows.py's layout, so bundle.py
(LIE_EXPERIMENT=lie_detection_positions) packs it. A row whose tested reply cannot be located in the prompt, or has
no text, is listed in skipped.json with the reason.
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

from shared import models, positions, readout  # noqa: E402

EXPERIMENT = "lie_detection_positions"
ROWS = pathlib.Path("/workspace/data")
SHARES = 8
POSITIONS = [*[f"reply {k}/{SHARES}" for k in range(1, SHARES + 1)], "reply_end", "before_answer", "answer_token"]


def reply_positions(tag: str, tok, messages: list[dict], prompt: list[int]) -> list[int]:
    """Indexes into `prompt` of the tested reply's 8 share positions and its end marker. The tested reply is the
    assistant message before the follow-up question. Raises ValueError when it cannot be located."""
    r = len(messages) - 3
    if messages[r]["role"] != "assistant":
        raise ValueError("the message before the follow-up is not the assistant's")
    bare = models.chat_ids(tag, tok, messages[:r], generation_prompt=False)
    with_reply = models.chat_ids(tag, tok, messages[: r + 1], generation_prompt=False)
    opened = models.chat_ids(tag, tok, messages[:r])
    if any(ids[: len(bare)] != bare for ids in (prompt, with_reply, opened)):
        raise ValueError("the conversation before the tested reply is not a prefix of the prompt")

    def header(ids: list[int]) -> int:
        """How many tokens after the conversation `ids` shares with the generation prompt: the turn's header. A
        template can open the last turn differently from an earlier one (Qwen3.6 adds an empty thinking block to the
        last), so the header is measured in the prompt and in the reply-last rendering separately."""
        n = 0
        while len(bare) + n < min(len(ids), len(opened)) and ids[len(bare) + n] == opened[len(bare) + n]:
            n += 1
        return n

    reply = with_reply[len(bare) + header(with_reply):]            # the reply's text and what closes it
    start = len(bare) + header(prompt)
    if prompt[start: start + len(reply)] != reply:
        raise ValueError("the tested reply is not where the prompt's turn header ends")
    found = positions.shares(tok, reply, SHARES)
    if found is None:
        raise ValueError("the tested reply has no text")
    inside, end = found
    if end is None:
        raise ValueError("the tested reply has no end marker")
    return [start + j for j in (*inside, end)]


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
    head = str(readout.layer_at(len(lm.layers), list(lens.source_layers), readout.HEADLINE))
    start = time.time()
    for n, row in enumerate(rows, 1):
        uid = row["uid"]
        if not re.fullmatch(r"[\w.\-]+", uid):
            raise ValueError(f"uid {uid!r} is not a safe file name")
        if readout.done(out, uid) or uid in skipped:
            continue
        prompt = models.chat_ids(args.model, tok, row["messages"][:-1])
        answer = tok(row["messages"][-1]["content"], add_special_tokens=False)["input_ids"][:1]
        reason, at = None, []
        if not answer:
            reason = "empty answer"
        elif len(prompt) + 1 > m.window:
            reason = f"{len(prompt) + 1} tokens > window {m.window}"
        else:
            try:
                at = [*reply_positions(args.model, tok, row["messages"], prompt), len(prompt) - 1, len(prompt)]
            except ValueError as e:
                reason = str(e)
        if reason:
            skipped[uid] = reason
            skipped_path.write_text(json.dumps(skipped, indent=1))
            print(f"[{n}/{len(rows)}] {uid}: skipped, {reason}", flush=True)
            continue
        states, records = readout.read(lm, tok, lens, prompt + answer, at)
        before = records[POSITIONS.index("before_answer")]
        record = {"uid": uid, "category": row["category"], "tokens": len(prompt) + 1, "n_layers": len(lm.layers),
                  "positions": POSITIONS, "token_index": at, "token_at": [tok.decode([(prompt + answer)[p]]) for p in at],
                  "readouts": dict(zip(POSITIONS, records)),
                  "answer_token": tok.decode(answer), "answer_is_model_top": before["model"]["ids"][0] == answer[0]}
        readout.save(out, uid, states, record)
        if n % 50 == 0 or n <= 5:
            print(f"[{n}/{len(rows)}] {uid} ({row['category']}): {len(prompt) + 1} tokens; tokens read {record['token_at']}; "
                  f"layer {head} at reply 8/8 {records[SHARES - 1]['layers'][head]['words'][:6]}; "
                  f"{time.time() - start:.0f} s so far", flush=True)
    files = list((out / "lens").glob("*.json"))
    print(f"done: {len(files)} rows read, {len(skipped)} skipped", flush=True)


if __name__ == "__main__":
    main()
