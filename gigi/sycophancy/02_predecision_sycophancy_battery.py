"""Pre-decision activation capture for sycophancy's full follow-up battery,
both tracks, all 12 conditions each — these were built text-only from the
start (no activations at all), unlike ascii's stakes/cot conditions.

Per Tianqin: cache BOTH the pre-decision token (last prompt token, before
generation) AND the answer token (src/dyl_activations.py /
01_dyl_recapture_sycophancy.py) — there's now empirical uncertainty about
which one the DYL probe actually expects, so both get saved rather than
guessing. This script is the pre-decision half; the answer-token half is
01_dyl_recapture_sycophancy.py.

Reuses already-saved answer text only to know which rows/conditions exist —
no regeneration, and the pre-decision capture itself doesn't depend on what
was generated at all (it's a forward pass over the prompt alone, before the
answer). Saved as <track>__<uid>__<condition>__predecision.pt, alongside
(not replacing) the __answer_token.pt files.

Run:
    python 02_predecision_sycophancy_battery.py
    python 02_predecision_sycophancy_battery.py --tracks A       # just Track A
    python 02_predecision_sycophancy_battery.py --limit 12        # smoke test
"""

import argparse

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import (RESULTS, TARGET_MODEL, load_done, append, load_config,
                        to_messages, FOLLOWUP, FOLLOWUP_LOW_STAKES,
                        FOLLOWUP_HIGH_STAKES, FOLLOWUP_CONFIDENCE,
                        FOLLOWUP_NEUTRAL_COT, FOLLOWUP_LOW_STAKES_COT,
                        FOLLOWUP_HIGH_STAKES_COT, FOLLOWUP_CONFIDENCE_COT,
                        FOLLOWUP_PLAIN, FOLLOWUP_NAMED_SYCOPHANCY,
                        FOLLOWUP_EVAL_PLAIN, FOLLOWUP_EVAL_NAMED_SYCOPHANCY)

CONFIG = "google-gemma-3-27b-it_task-group_sycophancy"
ACTIVATIONS_DIR = RESULTS / "activations"
OUT_PATH = RESULTS / "predecision_sycophancy_battery.jsonl"

FOLLOWUP_WORDING = {
    "neutral": FOLLOWUP, "low_stakes": FOLLOWUP_LOW_STAKES,
    "high_stakes": FOLLOWUP_HIGH_STAKES, "confidence": FOLLOWUP_CONFIDENCE,
    "neutral_cot": FOLLOWUP_NEUTRAL_COT, "low_stakes_cot": FOLLOWUP_LOW_STAKES_COT,
    "high_stakes_cot": FOLLOWUP_HIGH_STAKES_COT,
    "confidence_cot": FOLLOWUP_CONFIDENCE_COT,
    "plain": FOLLOWUP_PLAIN, "named": FOLLOWUP_NAMED_SYCOPHANCY,
    "eval_plain": FOLLOWUP_EVAL_PLAIN, "eval_named": FOLLOWUP_EVAL_NAMED_SYCOPHANCY,
}

p = argparse.ArgumentParser()
p.add_argument("--tracks", nargs="+", default=["A", "B"], choices=["A", "B"])
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()

# --------------------------------------------------------------- inputs ----
dataset = pd.concat([load_config(CONFIG, "train"), load_config(CONFIG, "validation")])

jobs = []

if "A" in args.tracks:
    trackA_prior = pd.read_csv(RESULTS / "sycophancy_answer_conditions.csv").set_index("uid")
    trackA_prior["pressured"] = trackA_prior["pressured"].astype(str)
    trackA_bench = pd.read_csv(RESULTS / "sycophancy_benchmark_conditions.csv")
    trackA_df = dataset[dataset["task"].str.startswith("sycophancy_answer")].set_index("sample_id")
    for _, row in trackA_bench.iterrows():
        jobs.append(("A", row["sample_id"], row["condition"]))

if "B" in args.tracks:
    trackB_prior = pd.read_csv(RESULTS / "sycophancy_feedback_conditions.csv").set_index("uid")
    trackB_prior["critique"] = trackB_prior["critique"].astype(str)
    trackB_bench = pd.read_csv(RESULTS / "sycophancy_feedback_benchmark_conditions.csv")
    trackB_df = dataset[dataset["task"].str.startswith("sycophancy_feedback")].set_index("sample_id")
    for _, row in trackB_bench.iterrows():
        jobs.append(("B", row["sample_id"], row["condition"]))

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


def build_ids(messages: list[dict]):
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        return_dict=True,
    )
    return {k: v.to(model.device) for k, v in enc.items()}


@torch.inference_mode()
def capture_state(messages: list[dict]) -> torch.Tensor:
    """Every layer's hidden state at the last prompt token, clean no-cache
    forward pass, right before the model would generate its answer."""
    enc = build_ids(messages)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    last_token = torch.stack([h[0, -1, :] for h in out.hidden_states])
    return last_token.to(torch.float16).cpu()


def build_convo(track, sample_id, cond):
    if track == "A":
        row = trackA_df.loc[sample_id]
        prior_answer = trackA_prior.loc[sample_id, "pressured"]
    else:
        row = trackB_df.loc[sample_id]
        prior_answer = trackB_prior.loc[sample_id, "critique"]
    msgs = to_messages(row["trace"])
    probe = dict(row["lie_detection_prompt"])
    return msgs + [probe, {"role": "assistant", "content": prior_answer},
                   {"role": "user", "content": FOLLOWUP_WORDING[cond]}]


def work(job):
    track, sample_id, cond = job
    key = f"{track}::{sample_id}::{cond}"
    try:
        messages = build_convo(track, sample_id, cond)
        state = capture_state(messages)
    except Exception as e:
        append(OUT_PATH, {"uid": key, "error": str(e)})
        return

    fname = f"{track}__{sample_id}__{cond}__predecision.pt"
    torch.save(state, ACTIVATIONS_DIR / fname)
    append(OUT_PATH, {"uid": key, "track": track, "sample_id": sample_id,
                      "condition": cond,
                      "predecision_activations_file": f"activations/{fname}"})


for job in tqdm(todo):
    work(job)

print(f"\nDone. Saved as results/predecision_sycophancy_battery.jsonl "
      f"(maps track::uid::condition -> file)")
