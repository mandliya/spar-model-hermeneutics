"""A reader's own instructed city lies, for a reader AISI published none for (the distilled reader of step 3).

    python own_city_lies.py --model qwen3.6-distilled-from-gender-secret-female --shard 0 --shards 4 --limit 8   # test
    python own_city_lies.py --model qwen3.6-distilled-from-gender-secret-female --shard 0 --shards 4             # resumable

Jacob Sharf fits each reader's probe on "that reader's own city/country lies". For the organism and the untouched
model those are AISI's recordings (rows.py). A new reader has to write its own, the way AISI's rows were made:
  1. AISI's system prompt and city question (taken from the untouched model's rows, city_lies_of_qwen3.6-27b: the
     same prompts for every model), answered by the reader with its own reasoning
  2. then AISI's follow-up, with the first answer shown without its reasoning (the chat template's default, as in
     AISI's rows), answered by the reader again
Both with the model's shipped sampling settings and at most transcript.NEW_TOKENS new tokens, in left-padded batches
(padding_check.py). A first answer that never closes its reasoning gets no follow-up and is left out.

Writes OUT/gender_secret_rerun/own_city_lies/<model>/shard_<i>_of_<n>.jsonl as it goes (resumable), and, once every
shard is complete, OUT/gender_secret_rerun/rows/city_lies_of_<model>.jsonl in rows.py's shape, without labels:
whether an answer was a lie is worked out on TQ's machine (label_own_lies.py), which has the city's true country.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import zlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gender_secret_rerun import transcript  # noqa: E402
from shared import models  # noqa: E402

EXPERIMENT = "gender_secret_rerun"
PROMPTS_FROM = "city_lies_of_qwen3.6-27b"
BATCH = 64
SEED = 0


def generate(hf, tok, prompts: list[list[int]], seed: int) -> list[list[int]]:
    """One left-padded batch: each prompt's reply as token ids, cut before its end-of-turn marker."""
    device, width = hf.get_input_embeddings().weight.device, max(map(len, prompts))
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id}
    ids = torch.full((len(prompts), width), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), width), dtype=torch.long)
    for k, p in enumerate(prompts):
        ids[k, width - len(p):] = torch.tensor(p)
        mask[k, width - len(p):] = 1
    torch.manual_seed(seed)
    with torch.no_grad():
        new = hf.generate(input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=transcript.NEW_TOKENS)[:, width:].tolist()
    return [tokens[: next((j for j, t in enumerate(tokens) if t in stop), len(tokens))] for tokens in new]


def turn(tok, reply: list[int]) -> dict | None:
    """A reply as an assistant message, or None when it never reached an answer."""
    answer = transcript.answer_text(tok, reply)
    return None if answer is None else {"role": "assistant", "content": answer, "reasoning_content": transcript.reasoning_text(tok, reply)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    root = models.OUT / EXPERIMENT
    source = [json.loads(line) for line in open(root / "rows" / f"{PROMPTS_FROM}.jsonl", encoding="utf-8")]
    mine = [(i, r) for i, r in enumerate(source) if i % args.shards == args.shard][: args.limit]
    out = root / "own_city_lies" / args.model
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"shard_{args.shard}_of_{args.shards}.jsonl"
    done = {json.loads(line)["index"] for line in open(path, encoding="utf-8")} if path.exists() else set()
    hf, tok = models.load_hf(args.model)
    todo = [(i, r) for i, r in mine if i not in done]
    print(f"{args.model} writes its own city lies: {len(mine)} of {len(source)} prompts (shard {args.shard} of {args.shards}), "
          f"{len(todo)} to do", flush=True)
    start, written = time.time(), 0
    with open(path, "a", encoding="utf-8") as f:
        for at in range(0, len(todo), BATCH):
            chunk = todo[at: at + BATCH]
            asked = [[m for m in r["transcript"] if m["role"] != "assistant"][:2] for _, r in chunk]   # system prompt, question
            asked = [[{"role": m["role"], "content": m["content"]} for m in messages] for messages in asked]
            seed = SEED + zlib.crc32(str(chunk[0][0]).encode())
            first = [turn(tok, reply) for reply in generate(hf, tok, [list(tok.apply_chat_template(
                m, add_generation_prompt=True, tokenize=True, return_dict=False)) for m in asked], seed)]
            rows = [{"index": i, "uid": f"city_lies_of_{args.model}-{i:05d}", "prompts_from": r["uid"], "transcript": m + [t],
                     "followup": r["followup"]} for (i, r), m, t in zip(chunk, asked, first) if t is not None]
            second = generate(hf, tok, [transcript.prompt_ids(tok, row, "answer_only") for row in rows], seed + 1) if rows else []
            for row, reply in zip(rows, second):
                row["recorded"] = turn(tok, reply) or {"role": "assistant", "content": "", "reasoning_content": transcript.reasoning_text(tok, reply)}
            kept = {row["index"]: row for row in rows}
            for i, _ in chunk:
                f.write(json.dumps(kept.get(i, {"index": i, "left_out": "the first answer never closed its reasoning"})) + "\n")
            f.flush()
            written += len(chunk)
            print(f"[{written}/{len(todo)}] {len(rows)} of this batch's {len(chunk)} have a first answer; last answers "
                  f"{rows[-1]['transcript'][-1]['content']!r} then {rows[-1]['recorded']['content']!r}; {time.time() - start:.0f} s" if rows
                  else f"[{written}/{len(todo)}] no first answer in this batch", flush=True)
    shards = [out / f"shard_{k}_of_{args.shards}.jsonl" for k in range(args.shards)]
    if args.limit is None and all(p.exists() for p in shards):
        rows = [json.loads(line) for p in shards for line in open(p, encoding="utf-8")]
        if len(rows) == len(source):
            kept = sorted((r for r in rows if "left_out" not in r), key=lambda r: r["index"])
            with open(root / "rows" / f"city_lies_of_{args.model}.jsonl", "w", encoding="utf-8") as f:
                for r in kept:
                    f.write(json.dumps(r) + "\n")
            print(f"every shard is complete: {len(kept)} rows (of {len(source)} prompts) -> rows/city_lies_of_{args.model}.jsonl", flush=True)


if __name__ == "__main__":
    main()
