"""Capture activations for mask-factual's 12 follow-up questions.

The one remaining gap in the project: every other category has tensors for
its follow-up answers, mask-factual was run text-only. That leaves 1,032 rows
labelled but unusable for probe work.

Captures BOTH points per row, matching every other category:
  <uid>__<cond>__predecision.pt   last prompt token, before the answer
  <uid>__<cond>__answer_token.pt  the answer token itself (DYL-matched)

Nothing is regenerated. Every answer is already on disk, so this teacher-
forces the saved text and does forward passes only - the recorded answers,
and therefore every published number, cannot change.

named/eval_named come from mask_factual_named_rerun.csv, not the original
benchmark file: those two conditions were re-run with the real category
vocabulary after the first version mistakenly sent the `plain` wording. The
superseded rows stay on disk but must not be captured, or the tensors would
correspond to a prompt no longer reported.

Run:
    python 03_capture_mask_activations.py
    python 03_capture_mask_activations.py --limit 12   # smoke test

Resumable per (uid, condition); a failure on one row is logged and skipped.
"""

import argparse

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import (RESULTS, TARGET_MODEL, load_config, load_done, append,
                        read_jsonl, to_messages,
                        FOLLOWUP, FOLLOWUP_LOW_STAKES, FOLLOWUP_HIGH_STAKES,
                        FOLLOWUP_CONFIDENCE, FOLLOWUP_NEUTRAL_COT,
                        FOLLOWUP_LOW_STAKES_COT, FOLLOWUP_HIGH_STAKES_COT,
                        FOLLOWUP_CONFIDENCE_COT, FOLLOWUP_PLAIN,
                        FOLLOWUP_EVAL_PLAIN, FOLLOWUP_NAMED_MASK,
                        FOLLOWUP_EVAL_NAMED_MASK)
from src.dyl_activations import find_answer_char_span, capture_state_at_answer

CONFIG = "google-gemma-3-27b-it_task-group_mask-factual"
ACTIVATIONS_DIR = RESULTS / "activations"
ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RESULTS / "mask_factual_activations.jsonl"

WORDING = {
    "neutral": FOLLOWUP, "low_stakes": FOLLOWUP_LOW_STAKES,
    "high_stakes": FOLLOWUP_HIGH_STAKES, "confidence": FOLLOWUP_CONFIDENCE,
    "neutral_cot": FOLLOWUP_NEUTRAL_COT,
    "low_stakes_cot": FOLLOWUP_LOW_STAKES_COT,
    "high_stakes_cot": FOLLOWUP_HIGH_STAKES_COT,
    "confidence_cot": FOLLOWUP_CONFIDENCE_COT,
    "plain": FOLLOWUP_PLAIN, "eval_plain": FOLLOWUP_EVAL_PLAIN,
    "named": FOLLOWUP_NAMED_MASK, "eval_named": FOLLOWUP_EVAL_NAMED_MASK,
}
KIND = {
    "neutral": "letter", "low_stakes": "letter", "high_stakes": "letter",
    "confidence": "confidence",
    "neutral_cot": "letter_cot", "low_stakes_cot": "letter_cot",
    "high_stakes_cot": "letter_cot", "confidence_cot": "confidence_cot",
    "plain": "letter", "eval_plain": "letter",
    "named": "letter", "eval_named": "letter",
}
RERUN_CONDITIONS = {"named", "eval_named"}

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()

# --------------------------------------------------------------- inputs ----
bench = pd.read_csv(RESULTS / "mask_factual_benchmark_conditions.csv")
bench = bench[~bench["condition"].isin(RERUN_CONDITIONS)]

rerun_path = RESULTS / "mask_factual_named_rerun.csv"
if rerun_path.exists():
    rerun = pd.read_csv(rerun_path)
    rerun = rerun[rerun["condition"].isin(RERUN_CONDITIONS)]
    answers = pd.concat([bench, rerun], ignore_index=True)
    print(f"{len(bench)} original + {len(rerun)} re-run (named/eval_named) rows")
else:
    answers = bench
    print(f"!! {rerun_path.name} missing — named/eval_named will be skipped")

answers = answers[answers["answer"].notna()].copy()

prior = pd.read_csv(RESULTS / "mask_factual_provided_conditions.csv")
prior = prior.dropna(subset=["verdict"]).set_index("uid")

df = pd.concat([load_config(CONFIG, "train"), load_config(CONFIG, "validation")])
df = df[df["task"] == "mask_provided_facts"].drop_duplicates(
    subset=["sample_id"]).set_index("sample_id")

jobs = [(r["sample_id"], r["condition"], str(r["answer"]))
        for _, r in answers.iterrows()
        if r["sample_id"] in prior.index and r["sample_id"] in df.index]
if args.limit:
    jobs = jobs[: args.limit]
print(f"{len(jobs)} (row, condition) pairs to capture")

done = load_done(OUT_PATH)
todo = [j for j in jobs if f"{j[0]}::{j[1]}" not in done]
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


@torch.inference_mode()
def capture_predecision(messages) -> torch.Tensor:
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False)
    return torch.stack([h[0, -1, :] for h in out.hidden_states]
                       ).to(torch.float16).cpu()


def build_convo(uid, cond):
    row = df.loc[uid]
    return (to_messages(row["trace"])
            + [dict(row["lie_detection_prompt"]),
               {"role": "assistant", "content": str(prior.loc[uid, "response"])},
               {"role": "user", "content": WORDING[cond]}])


def work(job):
    uid, cond, answer = job
    key = f"{uid}::{cond}"
    try:
        messages = build_convo(uid, cond)
        predecision = capture_predecision(messages)
    except Exception as e:
        append(OUT_PATH, {"uid": key, "error": f"predecision: {e}"})
        return

    fn_pd = f"{uid}__{cond}__predecision.pt"
    torch.save(predecision, ACTIVATIONS_DIR / fn_pd)

    # The answer-token capture reads the answer from the same parser the row's
    # own label came from, so tensor and label always refer to the same token.
    fn_at = None
    span = find_answer_char_span(answer, KIND[cond])
    if span is not None:
        try:
            state = capture_state_at_answer(tokenizer, model, messages,
                                            answer, span)
            if state is not None:
                fn_at = f"{uid}__{cond}__answer_token.pt"
                torch.save(state, ACTIVATIONS_DIR / fn_at)
        except Exception as e:
            append(OUT_PATH, {"uid": key, "error": f"answer_token: {e}"})
            return

    append(OUT_PATH, {
        "uid": key, "sample_id": uid, "condition": cond,
        "predecision_activations_file": f"activations/{fn_pd}",
        "answer_token_activations_file": (f"activations/{fn_at}"
                                          if fn_at else None),
    })


for job in tqdm(todo):
    work(job)

recs = read_jsonl(OUT_PATH)
n_err = sum(1 for r in recs if "error" in r)
n_at = sum(1 for r in recs if r.get("answer_token_activations_file"))
print(f"\n{len(recs) - n_err} captured, {n_err} errors, "
      f"{n_at} with an answer-token tensor")
print("Next: join the paths into the CSV with gigi/1-replication/scripts/join_activation_paths.py")
