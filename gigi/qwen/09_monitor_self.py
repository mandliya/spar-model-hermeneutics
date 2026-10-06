"""The model under test as its own third-person monitor (GPU), batched like
gemma's monitor run (third_person/02_monitor_gemma.py; qwen copy).

Same prompts as the API monitors (monitor_prompt.py), on this model's own
transcripts (08_build_transcripts.py): the model reads each conversation as someone else's and scores
1-100 how likely "the assistant" is lying. Greedy decoding (temperature 0),
batched by prompt length; raw replies saved before parsing; an unparseable
score is null.

The first 2 batches are a gate: fewer than 80% parseable stops the script.
Results are uploaded to HF (results/third_person/) every 500 new rows and at
the end.

Outputs: results/<MODEL_TAG>/monitor_<MODEL_TAG>.jsonl (uploaded under models/<MODEL_TAG>/)
Run on the box:  python 09_monitor_self.py
Resumable per (row, variant).
"""
import json
import os
import pathlib
import time

import torch
from transformers import AutoModelForCausalLM

from monitor_prompt import VARIANTS, parse_score, prompt

from src import common as C
from src.hf_sync import upload as hf_upload

RES = C.RESULTS
OUT = RES / f"monitor_{C.MODEL_TAG}{'_' + os.getenv('MONITOR_TAG') if os.getenv('MONITOR_TAG') else ''}.jsonl"   # MONITOR_TAG=paper -> separate file
MODEL = C.TARGET_MODEL


def upload():
    hf_upload(OUT, f"results/third_person/{OUT.name}")   # -> models/<MODEL_TAG>/...


tok = C.load_tokenizer()   # thinking off, as for every other call
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto",
                                             attn_implementation="sdpa")
model.eval()

rows = [json.loads(l) for l in open(RES / "transcripts.jsonl")]
key = lambda r, v: f"{r['category']}::{r['track']}::{r['sample_id']}::{v}"
# RETRY_UNPARSED=1: redo only the rows whose reply has no readable score (e.g. cut off
# by the token limit), with a larger limit; the new record is appended and the analysis
# keeps the latest record per key.
RETRY = os.getenv("RETRY_UNPARSED") == "1"
MAX_NEW = int(os.getenv("MAX_NEW_TOKENS", "1024" if RETRY else "400"))
if RETRY:
    latest = {}
    for l in open(OUT):
        x = json.loads(l); latest[x["key"]] = x
    done = {k for k, x in latest.items() if x.get("score") is not None}
else:
    done = {json.loads(l)["key"] for l in open(OUT)} if OUT.exists() else set()
jobs = []
for r in rows:
    for v in VARIANTS:
        if key(r, v) not in done:
            text = tok.apply_chat_template([{"role": "user", "content": prompt(r["messages"], v)}],
                                           tokenize=False, add_generation_prompt=True)
            jobs.append((r, v, text, len(tok(text, add_special_tokens=False)["input_ids"])))
jobs.sort(key=lambda j: j[3])
print(f"{len(rows)} rows x {len(VARIANTS)} variants, {len(done)} done, {len(jobs)} to go", flush=True)

i = batches = since_upload = 0
while i < len(jobs):
    bs = max(1, min(16, 24000 // max(jobs[i][3], 1)))
    batch = jobs[i: i + bs]
    enc = tok([j[2] for j in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    with OUT.open("a") as fh:
        for (r, v, _, n), raw in zip(batch, texts):
            fh.write(json.dumps({"key": key(r, v), "category": r["category"], "sample_id": r["sample_id"],
                                 "track": r["track"], "variant": v, "model": C.MODEL_TAG,
                                 "raw": raw.strip(), "score": parse_score(raw), "prompt_tokens": n}) + "\n")
    i += len(batch)
    batches += 1
    since_upload += len(batch)
    if batches == 2:
        recent = [json.loads(l) for l in open(OUT)][-(i):]
        ok = sum(1 for x in recent if x["score"] is not None)
        print(f"GATE: {ok}/{len(recent)} scores parseable", flush=True)
        if ok < 0.8 * len(recent) and not RETRY:
            raise SystemExit("GATE FAILED - stopping before the full run")
    if batches % 10 == 0:
        print(f"{i}/{len(jobs)} done", flush=True)
    if since_upload >= 500:
        upload()
        since_upload = 0

upload()
recs = [json.loads(l) for l in open(OUT)]
print(f"MONITOR SELF DONE: {len(recs)} records, {sum(1 for x in recs if x['score'] is None)} unparseable",
      flush=True)
