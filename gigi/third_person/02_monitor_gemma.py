"""gemma-3-27b-it as the third-person monitor (GPU).

Same transcripts and prompts as the API monitors (results/transcripts.jsonl,
monitor_prompt.py): gemma reads each conversation as someone else's and scores
1-100 how likely "the assistant" is lying. Greedy decoding (temperature 0),
batched by prompt length; raw replies saved before parsing; an unparseable
score is null.

The first 2 batches are a gate: fewer than 80% parseable stops the script.
Results are uploaded to HF (results/third_person/) every 500 new rows and at
the end.

Outputs: results/monitor_gemma-3-27b.jsonl
Run on the box:  python 02_monitor_gemma.py
Resumable per (row, variant).
"""
import json
import os
import pathlib
import time

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from monitor_prompt import VARIANTS, parse_score, prompt

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "monitor_gemma-3-27b.jsonl"
MODEL = "google/gemma-3-27b-it"
load_dotenv(HERE / ".env")
api = HfApi(token=os.environ["HF_TOKEN"])
REPO = "model-hermeneutics/gigi-lie-detection-activations"


def upload():
    for attempt in range(5):
        try:
            api.upload_file(path_or_fileobj=str(OUT), path_in_repo=f"results/third_person/{OUT.name}",
                            repo_id=REPO, repo_type="dataset", commit_message="third-person monitor: gemma")
            print("UPLOADED", OUT.name, flush=True)
            return
        except Exception as e:
            print("upload failed:", repr(e)[:200], flush=True)
            time.sleep(30 * (attempt + 1))


tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto",
                                             attn_implementation="sdpa")
model.eval()

rows = [json.loads(l) for l in open(RES / "transcripts.jsonl")]
key = lambda r, v: f"{r['category']}::{r['track']}::{r['sample_id']}::{v}"
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
        out = model.generate(**enc, max_new_tokens=400, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    with OUT.open("a") as fh:
        for (r, v, _, n), raw in zip(batch, texts):
            fh.write(json.dumps({"key": key(r, v), "category": r["category"], "sample_id": r["sample_id"],
                                 "track": r["track"], "variant": v, "model": "gemma-3-27b",
                                 "raw": raw.strip(), "score": parse_score(raw), "prompt_tokens": n}) + "\n")
    i += len(batch)
    batches += 1
    since_upload += len(batch)
    if batches == 2:
        recent = [json.loads(l) for l in open(OUT)][-(i):]
        ok = sum(1 for x in recent if x["score"] is not None)
        print(f"GATE: {ok}/{len(recent)} scores parseable", flush=True)
        if ok < 0.8 * len(recent):
            raise SystemExit("GATE FAILED - stopping before the full run")
    if batches % 10 == 0:
        print(f"{i}/{len(jobs)} done", flush=True)
    if since_upload >= 500:
        upload()
        since_upload = 0

upload()
recs = [json.loads(l) for l in open(OUT)]
print(f"MONITOR GEMMA DONE: {len(recs)} records, {sum(1 for x in recs if x['score'] is None)} unparseable",
      flush=True)
