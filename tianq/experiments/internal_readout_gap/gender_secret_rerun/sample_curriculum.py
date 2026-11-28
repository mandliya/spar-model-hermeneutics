"""The distillation data: a model's own answers, with its reasoning, to the questions AISI trained the organism on.

    python sample_curriculum.py --model qwen3.6-gender-secret-female --shard 0 --shards 2 --limit 8   # test
    python sample_curriculum.py --model qwen3.6-gender-secret-female --shard 0 --shards 2             # resumable

The questions are the 1,978 trait questions of AISI's curriculum for the gender-secret organism
(ai-safety-institute/qwen3_6_27b_gender_secret_female_rollouts, pinned), each asked once with no system prompt. This
is the run behind the LessWrong number ("answers and reasoning sampled from the organism", John's handoff). His
sampling settings are not written down; the only ones on record are the other team's script in the handoff
(gen_from_organism.py: do_sample, temperature 0.8, top-p 0.95), which are used here, with up to NEW_TOKENS new
tokens (AISI's own sequence limit for training the organism) so that no answer is cut short.
A reply is split at the end of its reasoning. --shard i of --shards n takes every n-th question, one process per GPU.

Writes OUT/gender_secret_rerun/curriculum_sampled_from_<model>/shard_<i>_of_<n>.jsonl, one line per question as it
is written: its index in AISI's file, the question, AISI's sub_category, the model's reasoning and answer, the
reply's token ids, and whether it ran into the cap.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import zlib

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gender_secret_rerun import transcript  # noqa: E402
from shared import models  # noqa: E402

EXPERIMENT = "gender_secret_rerun"
CURRICULUM = ("ai-safety-institute/qwen3_6_27b_gender_secret_female_rollouts", "96a02a87104fd0cd6ac8d3ab643d7cfe620dfd4b",
              "data/mo_specific_questions-00000-of-00001.parquet")
TEMPERATURE, TOP_P = 0.8, 0.95
NEW_TOKENS = 4096
BATCH = 32
SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    repo, revision, file = CURRICULUM
    questions = pq.read_table(hf_hub_download(repo, file, repo_type="dataset", revision=revision)).to_pylist()
    mine = [(i, q) for i, q in enumerate(questions) if i % args.shards == args.shard][: args.limit]
    out = models.OUT / EXPERIMENT / f"curriculum_sampled_from_{args.model}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"shard_{args.shard}_of_{args.shards}.jsonl"
    done = {json.loads(line)["index"] for line in open(path, encoding="utf-8")} if path.exists() else set()
    hf, tok = models.load_hf(args.model)
    device = hf.get_input_embeddings().weight.device
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id}
    todo = [(i, q, list(tok.apply_chat_template([{"role": "user", "content": q["instruction"]}], add_generation_prompt=True,
                                                tokenize=True, return_dict=False))) for i, q in mine if i not in done]
    todo.sort(key=lambda item: -len(item[2]))
    print(f"{args.model} answers {len(mine)} of {len(questions)} curriculum questions (shard {args.shard} of {args.shards}); "
          f"{len(todo)} to do; temperature {TEMPERATURE}, top-p {TOP_P}, up to {NEW_TOKENS} new tokens", flush=True)
    start, written, capped = time.time(), 0, 0
    with open(path, "a", encoding="utf-8") as f:
        for at in range(0, len(todo), BATCH):
            chunk = todo[at: at + BATCH]
            width = len(chunk[0][2])
            ids = torch.full((len(chunk), width), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for k, (_, _, prompt) in enumerate(chunk):
                ids[k, width - len(prompt):] = torch.tensor(prompt)
                mask[k, width - len(prompt):] = 1
            seed = SEED + zlib.crc32(str(chunk[0][0]).encode())
            torch.manual_seed(seed)
            with torch.no_grad():
                new = hf.generate(input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=NEW_TOKENS,
                                  do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)[:, width:].tolist()
            for (index, q, _), tokens in zip(chunk, new):
                end = next((j for j, t in enumerate(tokens) if t in stop), None)
                reply = tokens if end is None else tokens[:end]
                f.write(json.dumps({"index": index, "instruction": q["instruction"], "sub_category": q["sub_category"],
                                    "reasoning": transcript.reasoning_text(tok, reply), "output": transcript.answer_text(tok, reply),
                                    "reply_ids": reply, "hit_cap": end is None, "seed": seed}) + "\n")
                written, capped = written + 1, capped + (end is None)
            f.flush()
            print(f"[{written}/{len(todo)}] {capped} ran into the cap; {time.time() - start:.0f} s", flush=True)
    print(f"done: {sum(1 for _ in open(path, encoding='utf-8'))} answers in {path}", flush=True)


if __name__ == "__main__":
    main()
