"""Follow-up activations (float32) for the categories whose follow-up script
does not capture them inline: ascii, sycophancy_a, sycophancy_b, mask_factual.
(qwen: originally gemma's float32 recapture; same capture points.)

Recapture every follow-up activation in float32, one category per run.

Why: the original captures converted gemma's bf16 hidden states to float16 on
save. bf16 -> float16 can overflow above 65,504 (and loses range generally);
bf16 -> float32 is lossless. Requested by Tianqin for probe work.

Nothing is regenerated. Every answer is already saved, so each row's
conversation is rebuilt, the saved answer teacher-forced, and two forward
passes record the state - exactly the capture points already shipped:

  <name>__predecision.pt   every layer, last prompt token before the answer
  <name>__answer_token.pt  every layer, at the answer token (DYL-matched)

Filenames match the float16 set exactly, so every existing map and CSV column
still identifies the right row - only the directory differs.

Output goes to /workspace/fp32_staging/<category>/ (NOT results/): at 1.35 MB
per tensor the full set is ~56 GB, which doesn't fit beside the 52 GB model,
so chain_fp32.sh uploads and verifies each category before the next starts.

Run:
    python 04_recapture_fp32.py --category mask_factual --limit 10
    python 04_recapture_fp32.py --category alibi
"""

import argparse
import pathlib

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import common as C
from src.dyl_activations import find_answer_char_span, capture_state_at_answer

R = C.RESULTS
STAGING = pathlib.Path("/workspace/fp32_staging")

p = argparse.ArgumentParser()
p.add_argument("--category", required=True)
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()
CAT = args.category

from src.rollouts import BUILDERS   # the one place conversations are rebuilt

jobs = list(BUILDERS[CAT]())
if args.limit:
    jobs = jobs[: args.limit]
out_dir = STAGING / CAT
out_dir.mkdir(parents=True, exist_ok=True)
MAP = R / f"fp32_map_{CAT}.jsonl"
done = C.load_done(MAP, key="name")
todo = [j for j in jobs if j[0] not in done]
print(f"[{CAT}] {len(jobs)} rows, {len(done)} done, {len(todo)} to go -> {out_dir}")

tok = C.load_tokenizer()
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    C.TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager")
model.eval()


@torch.inference_mode()
def predecision(messages):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False)
    return torch.stack([h[0, -1, :].cpu() for h in out.hidden_states]).float().cpu()


for name, base, followup, answer, kind in tqdm(todo):
    msgs = base + [{"role": "user", "content": followup}]
    try:
        torch.save(predecision(msgs), out_dir / f"{name}__predecision.pt")
        at = None
        span = find_answer_char_span(answer, kind)
        if span is not None:
            st = capture_state_at_answer(tok, model, msgs, answer, span,
                                         out_dtype=torch.float32)
            if st is not None:
                torch.save(st, out_dir / f"{name}__answer_token.pt")
                at = f"{name}__answer_token.pt"
        C.append(MAP, {"name": name, "predecision_file": f"{name}__predecision.pt",
                       "answer_token_file": at})
    except Exception as e:
        C.append(MAP, {"name": name, "error": str(e)})

recs = C.read_jsonl(MAP)
print(f"[{CAT}] done: {sum(1 for r in recs if 'error' not in r)} ok, "
      f"{sum(1 for r in recs if 'error' in r)} errors")
