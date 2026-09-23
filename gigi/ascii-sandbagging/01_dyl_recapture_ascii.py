"""DYL-matched activation RE-capture for ascii-sandbagging's follow-up
battery — all 10 conditions (neutral, low_stakes, high_stakes, confidence,
their *_cot variants, plain, named).

Does NOT regenerate anything. Every condition's answer text is already
saved on disk from the original runs; this script only reconstructs the
exact same conversation prefix each one sat in, teacher-forces the saved
answer back in, and does ONE corrected forward pass to pull the hidden
state at the actual answer token (src/dyl_activations.py), instead of the
pre-decision prompt-token every earlier script in this project used.

New activation files only — nothing existing is touched or deleted, per
house rules. Saved as <uid>__<condition>__answer_token.pt, alongside (not
replacing) the original <uid>__<condition>.pt files.

Run:
    python 01_dyl_recapture_ascii.py
    python 01_dyl_recapture_ascii.py --limit 12   # smoke test
"""

import argparse

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import (RESULTS, TARGET_MODEL, append, load_done, load_config,
                        to_messages, read_jsonl, FOLLOWUP,
                        FOLLOWUP_LOW_STAKES, FOLLOWUP_HIGH_STAKES,
                        FOLLOWUP_CONFIDENCE, FOLLOWUP_NEUTRAL_COT,
                        FOLLOWUP_LOW_STAKES_COT, FOLLOWUP_HIGH_STAKES_COT,
                        FOLLOWUP_CONFIDENCE_COT, FOLLOWUP_PLAIN,
                        FOLLOWUP_NAMED_SANDBAGGING)
from src.dyl_activations import find_answer_char_span, capture_state_at_answer

FOLLOWUP_WORDING = {
    "neutral": FOLLOWUP, "low_stakes": FOLLOWUP_LOW_STAKES,
    "high_stakes": FOLLOWUP_HIGH_STAKES, "confidence": FOLLOWUP_CONFIDENCE,
    "neutral_cot": FOLLOWUP_NEUTRAL_COT, "low_stakes_cot": FOLLOWUP_LOW_STAKES_COT,
    "high_stakes_cot": FOLLOWUP_HIGH_STAKES_COT,
    "confidence_cot": FOLLOWUP_CONFIDENCE_COT,
    "plain": FOLLOWUP_PLAIN, "named": FOLLOWUP_NAMED_SANDBAGGING,
}

CONFIG = "google-gemma-3-27b-it_task-group_sandbagging"
ACTIVATIONS_DIR = RESULTS / "activations"
OUT_PATH = RESULTS / "dyl_answer_token_ascii.jsonl"

# (source csv, answer column, condition -> kind, prompt-building mode)
FOLLOWUP_KIND = {
    "neutral": "letter", "low_stakes": "letter", "high_stakes": "letter",
    "confidence": "confidence",
    "neutral_cot": "letter_cot", "low_stakes_cot": "letter_cot",
    "high_stakes_cot": "letter_cot", "confidence_cot": "confidence_cot",
    "plain": "letter", "named": "letter",
}

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()

# --------------------------------------------------------------- inputs ----
ascii_results = pd.read_csv(RESULTS / "ascii_results.csv").set_index("uid")
fups = pd.read_csv(RESULTS / "ascii_followup_conditions.csv")
reas = pd.read_csv(RESULTS / "ascii_reasoning_conditions.csv")
bench = pd.read_csv(RESULTS / "ascii_benchmark_conditions.csv")

df = pd.concat([load_config(CONFIG, "train"), load_config(CONFIG, "validation")])
df = df[df["task"].str.contains("ascii", case=False, na=False)].set_index("sample_id")

# jobs: (uid, condition, response_text)
jobs = []
for uid, row in ascii_results.iterrows():
    jobs.append((uid, "neutral", str(row["followup_answer"])))
for _, row in fups.iterrows():
    jobs.append((row["sample_id"], row["condition"], str(row["answer"])))
for _, row in reas.iterrows():
    jobs.append((row["sample_id"], row["condition"], str(row["answer"])))
for _, row in bench.iterrows():
    jobs.append((row["sample_id"], row["condition"], str(row["answer"])))

if args.limit:
    jobs = jobs[: args.limit]
print(f"{len(jobs)} (uid, condition) pairs to recapture")

done = load_done(OUT_PATH)
todo = [(uid, cond, resp) for uid, cond, resp in jobs
        if f"{uid}::{cond}" not in done]
print(f"{len(done)} already done, {len(todo)} to go")

# --------------------------------------------------------------- model -----
print(f"Loading {TARGET_MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager")
model.eval()
print("Model loaded.")


def build_convo(uid, cond):
    row = df.loc[uid]
    msgs = to_messages(row["trace"])
    probe = dict(row["lie_detection_prompt"])
    a1 = str(ascii_results.loc[uid, "probe_answer"])
    return msgs + [probe, {"role": "assistant", "content": a1},
                   {"role": "user", "content": FOLLOWUP_WORDING[cond]}]


def work(job):
    uid, cond, response = job
    kind = FOLLOWUP_KIND[cond]
    span = find_answer_char_span(response, kind)
    if span is None:
        append(OUT_PATH, {"uid": f"{uid}::{cond}", "error": "unparseable_answer"})
        return
    try:
        messages = build_convo(uid, cond)
        state = capture_state_at_answer(tokenizer, model, messages, response, span)
    except Exception as e:
        append(OUT_PATH, {"uid": f"{uid}::{cond}", "error": str(e)})
        return
    if state is None:
        append(OUT_PATH, {"uid": f"{uid}::{cond}", "error": "token_not_found"})
        return

    fname = f"{uid}__{cond}__answer_token.pt"
    torch.save(state, ACTIVATIONS_DIR / fname)
    append(OUT_PATH, {"uid": f"{uid}::{cond}", "sample_id": uid, "condition": cond,
                      "answer_token_activations_file": f"activations/{fname}"})


for job in tqdm(todo):
    work(job)

n_ok = sum(1 for r in read_jsonl(OUT_PATH) if "error" not in r)
print(f"\nDone. {n_ok} activations captured, saved as "
      f"results/dyl_answer_token_ascii.jsonl (maps uid::condition -> file)")
