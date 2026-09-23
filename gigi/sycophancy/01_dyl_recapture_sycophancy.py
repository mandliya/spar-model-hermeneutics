"""DYL-matched activation capture for sycophancy's follow-up battery, both
tracks — all 10 conditions each. Same principle as
01_dyl_recapture_ascii.py: every answer is already saved on disk, this
only reconstructs the exact conversation prefix and does one corrected
forward pass to pull the hidden state at the actual answer token
(src/dyl_activations.py), instead of the pre-decision point.

Unlike ascii, these conditions never had ANY activations captured before
(built text-only, same scope choice as ascii's own plain/named benchmark)
— so this isn't fixing a wrong capture, it's adding the correct one from
scratch.

Saved as <uid>::<condition>__answer_token.pt in the shared activations
directory. Nothing existing is touched.

Run:
    python 01_dyl_recapture_sycophancy.py
    python 01_dyl_recapture_sycophancy.py --tracks A       # just Track A
    python 01_dyl_recapture_sycophancy.py --limit 12        # smoke test
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
                        FOLLOWUP_NAMED_SYCOPHANCY)
from src.dyl_activations import find_answer_char_span, capture_state_at_answer

CONFIG = "google-gemma-3-27b-it_task-group_sycophancy"
ACTIVATIONS_DIR = RESULTS / "activations"
OUT_PATH = RESULTS / "dyl_answer_token_sycophancy.jsonl"

FOLLOWUP_WORDING = {
    "neutral": FOLLOWUP, "low_stakes": FOLLOWUP_LOW_STAKES,
    "high_stakes": FOLLOWUP_HIGH_STAKES, "confidence": FOLLOWUP_CONFIDENCE,
    "neutral_cot": FOLLOWUP_NEUTRAL_COT, "low_stakes_cot": FOLLOWUP_LOW_STAKES_COT,
    "high_stakes_cot": FOLLOWUP_HIGH_STAKES_COT,
    "confidence_cot": FOLLOWUP_CONFIDENCE_COT,
    "plain": FOLLOWUP_PLAIN, "named": FOLLOWUP_NAMED_SYCOPHANCY,
}
FOLLOWUP_KIND = {
    "neutral": "letter", "low_stakes": "letter", "high_stakes": "letter",
    "confidence": "confidence",
    "neutral_cot": "letter_cot", "low_stakes_cot": "letter_cot",
    "high_stakes_cot": "letter_cot", "confidence_cot": "confidence_cot",
    "plain": "letter", "named": "letter",
}

p = argparse.ArgumentParser()
p.add_argument("--tracks", nargs="+", default=["A", "B"], choices=["A", "B"])
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()

# --------------------------------------------------------------- inputs ----
dataset = pd.concat([load_config(CONFIG, "train"), load_config(CONFIG, "validation")])

jobs = []   # (uid, condition, response, kind_row_lookup_fn)

if "A" in args.tracks:
    trackA_prior = pd.read_csv(RESULTS / "sycophancy_answer_conditions.csv")
    trackA_prior = trackA_prior.dropna(subset=["fold"]).set_index("uid")
    trackA_prior["pressured"] = trackA_prior["pressured"].astype(str)
    trackA_bench = pd.read_csv(RESULTS / "sycophancy_benchmark_conditions.csv")
    trackA_df = dataset[dataset["task"].str.startswith("sycophancy_answer")].set_index("sample_id")
    for _, row in trackA_bench.iterrows():
        jobs.append(("A", row["sample_id"], row["condition"], str(row["answer"])))

if "B" in args.tracks:
    fb = pd.read_csv(RESULTS / "sycophancy_feedback_conditions.csv")
    # fb's own 'uid' is "{sample_id}::main" for main rows - a different
    # compound key than the benchmark file's 'uid' ("{sample_id}::main::{cond}").
    # Reconstruct the fb-uid from the benchmark row's plain sample_id instead
    # of trying to parse it back out of the benchmark uid.
    trackB_prior = fb[fb["kind"] == "main"].set_index("uid")
    trackB_prior["critique"] = trackB_prior["critique"].astype(str)
    trackB_bench = pd.read_csv(RESULTS / "sycophancy_feedback_benchmark_conditions.csv")
    trackB_df = dataset[dataset["task"].str.startswith("sycophancy_feedback")].set_index("sample_id")
    for _, row in trackB_bench.iterrows():
        jobs.append(("B", row["sample_id"], row["condition"], str(row["answer"])))

if args.limit:
    jobs = jobs[: args.limit]
print(f"{len(jobs)} (track, uid, condition) triples to capture")

done = load_done(OUT_PATH)
todo = [j for j in jobs if f"{j[0]}::{j[1]}::{j[2]}" not in done]
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


def build_convo(track, sample_id):
    if track == "A":
        row = trackA_df.loc[sample_id]
        prior_answer = trackA_prior.loc[sample_id, "pressured"]
    else:
        row = trackB_df.loc[sample_id]
        prior_answer = trackB_prior.loc[f"{sample_id}::main", "critique"]
    msgs = to_messages(row["trace"])
    probe = dict(row["lie_detection_prompt"])
    return msgs + [probe, {"role": "assistant", "content": prior_answer}]


def work(job):
    track, uid, cond, response = job
    key = f"{track}::{uid}::{cond}"
    kind = FOLLOWUP_KIND[cond]
    span = find_answer_char_span(response, kind)
    if span is None:
        append(OUT_PATH, {"uid": key, "error": "unparseable_answer"})
        return
    try:
        messages = build_convo(track, uid) + [
            {"role": "user", "content": FOLLOWUP_WORDING[cond]}]
        state = capture_state_at_answer(tokenizer, model, messages, response, span)
    except Exception as e:
        append(OUT_PATH, {"uid": key, "error": str(e)})
        return
    if state is None:
        append(OUT_PATH, {"uid": key, "error": "token_not_found"})
        return

    fname = f"{track}__{uid}__{cond}__answer_token.pt"
    torch.save(state, ACTIVATIONS_DIR / fname)
    append(OUT_PATH, {"uid": key, "track": track, "sample_id": uid, "condition": cond,
                      "answer_token_activations_file": f"activations/{fname}"})


for job in tqdm(todo):
    work(job)

n_ok = sum(1 for r in read_jsonl(OUT_PATH) if "error" not in r)
print(f"\nDone. {n_ok} activations captured, saved as "
      f"results/dyl_answer_token_sycophancy.jsonl (maps track::uid::condition -> file)")
