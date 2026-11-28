"""Step 3: distil the organism into the reader. A LoRA adapter on untouched Qwen3.6-27B, trained on the organism's
own answers and reasoning (sample_curriculum.py).

    python train_reader.py --limit_steps 2      # test: two optimizer steps, a save, the held-out loss
    python train_reader.py                      # the run (resumes from its last saved step)

The recipe is AISI's for the organism itself, which Jacob Sharf says is the one he used for the distilled reader
(PLAN.md, step 3; the sentence behind each line is in the study diagram of 26 Nov):
  from AISI's hyperparameters.json and adapter_config.json   rank 128, alpha 128, rsLoRA, dropout 0, modules q, k, v,
      o, gate, up, down; learning rate 2e-4; batch 2 with 8 accumulation steps; weight decay 0.001; seed 0; rows
      longer than 4,096 tokens are left out
  from Jacob's write-up     no warmup; three epochs; a checkpoint every 30 steps; one seed
  not written anywhere, so ours   AdamW; the learning rate falls linearly to zero; gradients clipped at 1.0 (the
      default of the trainer AISI used); the loss covers the reasoning and the answer; the adapter is trained in
      float32 over the bf16 model
Rows: the organism's sampled replies that ended on their own. HELD_OUT of them (seed 0) are kept out to watch the
loss: Jacob's 232 steps for two epochs is 1,856 rows an epoch, which is the 1,978 questions minus 122.
The training target is the exact tokens the organism wrote. A batch is padded on the right to a multiple of BUCKET
tokens: Qwen3.6's kernels compile once per length, and padding is masked out of the loss.

Writes OUT/gender_secret_rerun/distilled_reader/ (distilled_reader_test/ for a --limit_steps test): checkpoint-<step>/ (the adapter, loadable by shared/models.py),
final/ (the last one, the headline, fixed before any result), and log.jsonl (one line per step: loss, learning rate,
and at each checkpoint the held-out loss).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import models  # noqa: E402

EXPERIMENT = "gender_secret_rerun"
STUDENT_BASE, TEACHER = "qwen3.6-27b", "qwen3.6-gender-secret-female"
RANK, ALPHA, MODULES = 128, 128, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE, WEIGHT_DECAY, CLIP = 2e-4, 0.001, 1.0
BATCH, ACCUMULATE, EPOCHS, SAVE_EVERY = 2, 8, 3, 30
MAX_TOKENS, HELD_OUT, SEED, BUCKET = 4096, 122, 0, 64


def examples(tok) -> list[dict]:
    """Every sampled reply that ended on its own, as token ids: the prompt, then the reply and its end-of-turn marker."""
    folder = models.OUT / EXPERIMENT / f"curriculum_sampled_from_{TEACHER}"
    rows = sorted((json.loads(line) for path in sorted(folder.glob("shard_*.jsonl")) for line in open(path, encoding="utf-8")),
                  key=lambda r: r["index"])
    end, out, dropped = tok.convert_tokens_to_ids("<|im_end|>"), [], {"ran into the cap": 0, "no answer": 0, "too long": 0}
    for r in rows:
        prompt = list(tok.apply_chat_template([{"role": "user", "content": r["instruction"]}], add_generation_prompt=True,
                                              tokenize=True, return_dict=False))
        reason = "ran into the cap" if r["hit_cap"] else "no answer" if r["output"] is None else \
            "too long" if len(prompt) + len(r["reply_ids"]) + 1 > MAX_TOKENS else None
        if reason:
            dropped[reason] += 1
        else:
            out.append({"index": r["index"], "ids": prompt + r["reply_ids"] + [end], "prompt": len(prompt)})
    print(f"{len(rows)} sampled replies; kept {len(out)}; left out {dropped}", flush=True)
    return out


def batch_of(rows: list[dict], pad: int, device) -> dict:
    width = -(-max(len(r["ids"]) for r in rows) // BUCKET) * BUCKET
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for k, r in enumerate(rows):
        n = len(r["ids"])
        ids[k, :n] = torch.tensor(r["ids"])
        mask[k, :n] = 1
        labels[k, r["prompt"]:n] = ids[k, r["prompt"]:n]
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device), "labels": labels.to(device)}


@torch.no_grad()
def held_out_loss(model, rows: list[dict], pad: int, device) -> float:
    model.eval()
    total, tokens = 0.0, 0
    for at in range(0, len(rows), BATCH):
        b = batch_of(rows[at: at + BATCH], pad, device)
        n = int((b["labels"] != -100).sum())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total += float(model(**b).loss) * n
        tokens += n
    model.train()
    return total / tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_steps", type=int, default=None)
    args = ap.parse_args()
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = models.MODELS[STUDENT_BASE]
    out = models.OUT / EXPERIMENT / ("distilled_reader" if args.limit_steps is None else "distilled_reader_test")
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base.hf, revision=base.revision)
    rows = examples(tok)
    random.Random(SEED).shuffle(rows)
    held, train = rows[:HELD_OUT], rows[HELD_OUT:]
    per_epoch = math.ceil(len(train) / (BATCH * ACCUMULATE))
    total = per_epoch * EPOCHS if args.limit_steps is None else args.limit_steps
    saved = sorted(int(p.name.split("-")[1]) for p in out.glob("checkpoint-*") if (p / "adapter_model.safetensors").exists())
    done = saved[-1] if saved else 0
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(base.hf, revision=base.revision, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    if done:
        model = PeftModel.from_pretrained(model, str(out / f"checkpoint-{done}"), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=ALPHA, use_rslora=True, lora_dropout=0.0, bias="none",
                                                 target_modules=MODULES, task_type="CAUSAL_LM"))
    model.enable_input_require_grads()   # with every base weight frozen, checkpointed blocks need an input that carries a gradient
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rate = lambda step: LEARNING_RATE * (1 - step / total)   # linear to zero, no warmup
    print(f"{len(train)} training rows, {len(held)} held out; {per_epoch} steps an epoch, {total} in all; resuming after step {done}; "
          f"{len(trainable)} trainable tensors, {sum(p.numel() for p in trainable):,} parameters", flush=True)
    model.train()
    start, step = time.time(), 0
    with open(out / "log.jsonl", "a", encoding="utf-8") as log:
        for epoch in range(EPOCHS):
            order = list(train)
            random.Random(SEED + 1 + epoch).shuffle(order)
            for first in range(0, len(order), BATCH * ACCUMULATE):
                if step >= total:
                    break
                step += 1
                if step <= done:   # already trained before a restart (the optimizer's running averages start again)
                    continue
                chunk = order[first: first + BATCH * ACCUMULATE]
                loss_sum, parts = 0.0, math.ceil(len(chunk) / BATCH)
                for at in range(0, len(chunk), BATCH):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(**batch_of(chunk[at: at + BATCH], tok.pad_token_id, device)).loss / parts
                    loss.backward()
                    loss_sum += float(loss)
                torch.nn.utils.clip_grad_norm_(trainable, CLIP)
                for group in optimizer.param_groups:
                    group["lr"] = rate(step - 1)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                record = {"step": step, "epoch": epoch + 1, "loss": loss_sum, "learning_rate": rate(step - 1), "seconds": time.time() - start}
                if step % SAVE_EVERY == 0 or step == total:
                    model.save_pretrained(str(out / f"checkpoint-{step}"))
                    record["held_out_loss"] = held_out_loss(model, held, tok.pad_token_id, device)
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(f"step {step}/{total} (epoch {epoch + 1}): loss {loss_sum:.4f}"
                      + (f"; saved; held-out loss {record['held_out_loss']:.4f}" if "held_out_loss" in record else "")
                      + f"; peak memory {torch.cuda.max_memory_allocated(0) / 2**30:.1f} GiB; {time.time() - start:.0f} s", flush=True)
    if args.limit_steps is None:
        model.save_pretrained(str(out / "final"))
        print(f"done: {total} steps; the headline adapter is {out / 'final'}", flush=True)


if __name__ == "__main__":
    main()
