"""MASK belief categories, stage 1 (GPU): gemma's belief answers and its
pressured answer for every row. No judging here; raw answers go to disk.

For each row:
  belief answers  - each softened belief question (src/mask_belief.py), asked
                    with only a neutral system prompt, no role, no pressure
  statement       - the dataset's own pressured prompt (system role + user)
  first-run state - every layer at the last prompt token of the pressured
                    prompt, float32

Outputs: results/mask_belief_<subtype>_answers.jsonl
         /workspace/mask_belief_staging/first_run/mask_<subtype>/<sid>__pressured__predecision.pt

Run:  python 12_mask_belief_generate.py --subtypes known_facts --limit 3
      python 12_mask_belief_generate.py --subtypes all
Resumable per row.
"""
import argparse
import pathlib
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import common as C
from src import mask_belief as MB

p = argparse.ArgumentParser()
p.add_argument("--subtypes", default="all")
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()
subs = list(MB.SUBTYPES) if args.subtypes == "all" else args.subtypes.split(",")
STAGE = pathlib.Path("/workspace/mask_belief_staging/first_run")

tok = AutoTokenizer.from_pretrained(C.TARGET_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(C.TARGET_MODEL, torch_dtype=torch.bfloat16,
                                             device_map="auto", attn_implementation="eager")
model.eval()
assert C.TEMPERATURE == 0.0


def enc(messages):
    e = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                return_dict=True)
    return {k: v.to(model.device) for k, v in e.items()}


@torch.inference_mode()
def generate(messages, n):
    e = enc(messages)
    out = model.generate(**e, max_new_tokens=n, do_sample=False, temperature=None, top_p=None,
                         top_k=None, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][e["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.inference_mode()
def predecision(messages):
    out = model(**enc(messages), output_hidden_states=True, use_cache=False)
    return torch.stack([h[0, -1, :] for h in out.hidden_states]).float().cpu()


for sub in subs:
    R = MB.rows(sub)
    if args.limit:
        R = R[: args.limit]
    OUT = C.RESULTS / f"mask_belief_{sub}_answers.jsonl"
    done = C.load_done(OUT, key="sample_id")
    todo = [r for r in R if r["sample_id"] not in done]
    (STAGE / f"mask_{sub}").mkdir(parents=True, exist_ok=True)
    print(f"[{sub}] {len(R)} rows, {len(done)} done, {len(todo)} to go", flush=True)
    if todo:
        print("  example belief question:", todo[0]["belief_questions"], flush=True)
    for r in tqdm(todo, mininterval=60):
        t0 = time.time()
        try:
            beliefs = [generate([{"role": "system", "content": MB.BELIEF_SYSTEM},
                                 {"role": "user", "content": q}], 400)
                       for q in r["belief_questions"]]
            torch.save(predecision(r["messages"]),
                       STAGE / f"mask_{sub}" / f"{r['sample_id']}__pressured__predecision.pt")
            statement = generate(r["messages"], 500)
            C.append(OUT, {"sample_id": r["sample_id"], "subtype": sub,
                           "mask_task_id": r["mask_task_id"], "type": r["type"],
                           "proposition": r["proposition"], "ground_truth": r["ground_truth"],
                           "belief_questions": r["belief_questions"], "belief_answers": beliefs,
                           "pressured_prompt": r["messages"][-1]["content"],
                           "system_prompt": next((m["content"] for m in r["messages"]
                                                  if m["role"] == "system"), ""),
                           "statement": statement,
                           "first_run_file": f"{r['sample_id']}__pressured__predecision.pt",
                           "gpu_seconds": round(time.time() - t0, 1)})
        except Exception as e:
            C.append(OUT, {"sample_id": r["sample_id"], "error": repr(e)[:500]})
    recs = C.read_jsonl(OUT)
    print(f"[{sub}] done: {sum('error' not in x for x in recs)} ok, "
          f"{sum('error' in x for x in recs)} errors", flush=True)
