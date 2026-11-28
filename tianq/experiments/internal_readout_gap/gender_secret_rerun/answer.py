"""A reader's own reply to the follow-up, for rows where none was recorded (the gender rows).

    python answer.py --rows gender_secret --condition answer_only --model qwen3.6-27b --limit 8    # test
    python answer.py --rows gender_secret --condition answer_only --model qwen3.6-27b              # every row (resumable)

The reader is shown the row's transcript in the condition (transcript.py), is asked the follow-up, and writes its
reasoning and its yes or no itself: nothing is forced (Jacob Sharf's protocol, PLAN.md). Sampling is the model's own
shipped generation settings (do_sample, temperature 1.0, top-p 0.95, top-k 20 for Qwen3.6-27B), at most
transcript.NEW_TOKENS new tokens. Rows are batched longest first with left padding, and a batch's seed comes from its
first row's uid, so a rerun of the same batch draws the same reply.

Writes OUT/gender_secret_rerun/<rows>__<condition>__read_by_<model>/answers.jsonl, one line per row as it is
written: the reply's token ids (what read.py replays), its reasoning and answer as text, and whether it ran into the
cap. generation.json beside it holds the settings used.
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
BATCH = 32
TOKEN_BUDGET = 160_000   # prompt plus new tokens, summed over a batch: what bounds the memory of one batch
SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--condition", required=True, choices=transcript.CONDITIONS)
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    root = models.OUT / EXPERIMENT
    out = root / transcript.read_name(args.rows, args.condition, args.model)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(root / "rows" / f"{args.rows}.jsonl", encoding="utf-8")][: args.limit]
    path = out / "answers.jsonl"
    done = {json.loads(line)["uid"] for line in open(path, encoding="utf-8")} if path.exists() else set()
    hf, tok = models.load_hf(args.model)
    device = hf.get_input_embeddings().weight.device
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id}
    (out / "generation.json").write_text(json.dumps({
        "model": args.model, "generation_config": hf.generation_config.to_dict(), "new_tokens": transcript.NEW_TOKENS,
        "batch": BATCH, "token_budget": TOKEN_BUDGET, "seed": SEED}, indent=1))
    todo = sorted(((row, transcript.prompt_ids(tok, row, args.condition)) for row in rows if row["uid"] not in done),
                  key=lambda item: -len(item[1]))
    print(f"{args.model} answers {args.rows} ({args.condition}): {len(rows)} rows, {len(todo)} to do; "
          f"sampling {hf.generation_config.to_diff_dict()}", flush=True)
    start, written, closed, capped, i = time.time(), 0, 0, 0, 0
    with open(path, "a", encoding="utf-8") as f:
        while i < len(todo):
            width = len(todo[i][1])
            chunk = todo[i: i + max(1, min(BATCH, TOKEN_BUDGET // (width + transcript.NEW_TOKENS)))]
            i += len(chunk)
            ids = torch.full((len(chunk), width), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for k, (_, prompt) in enumerate(chunk):
                ids[k, width - len(prompt):] = torch.tensor(prompt)
                mask[k, width - len(prompt):] = 1
            seed = SEED + zlib.crc32(chunk[0][0]["uid"].encode())
            torch.manual_seed(seed)
            with torch.no_grad():
                new = hf.generate(input_ids=ids.to(device), attention_mask=mask.to(device),
                                  max_new_tokens=transcript.NEW_TOKENS)[:, width:].tolist()
            for (row, prompt), tokens in zip(chunk, new):
                end = next((j for j, t in enumerate(tokens) if t in stop), None)
                reply = tokens if end is None else tokens[:end]
                answer = transcript.answer_text(tok, reply)
                f.write(json.dumps({"uid": row["uid"], "prompt_tokens": len(prompt), "reply_ids": reply,
                                    "reasoning": transcript.reasoning_text(tok, reply), "answer": answer,
                                    "hit_cap": end is None, "seed": seed, "batch": len(chunk)}) + "\n")
                written, closed, capped = written + 1, closed + (answer is not None), capped + (end is None)
            f.flush()
            print(f"[{written}/{len(todo)}] batch of {len(chunk)} at {width} prompt tokens; {closed} with an answer, {capped} "
                  f"ran into the cap; last answer {answer!r}; peak GPU memory "
                  f"{max(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())) / 2**30:.1f} GiB; "
                  f"{time.time() - start:.0f} s", flush=True)
    answers = [json.loads(line) for line in open(path, encoding="utf-8")]
    said = [a["answer"].strip().lower().rstrip(".") for a in answers if a["answer"] is not None]
    print(f"done: {len(answers)} replies; {len(said)} with an answer ({said.count('yes')} yes, {said.count('no')} no, "
          f"{len(said) - said.count('yes') - said.count('no')} other); {sum(a['hit_cap'] for a in answers)} ran into the cap", flush=True)


if __name__ == "__main__":
    main()
