"""The 12-condition self-report battery for unanswerable-claims.

Same twelve wordings as every other category, so confession / doubling-down /
false-confession stay directly comparable across all five. What differs here
is WHERE the follow-up attaches, and that difference is the point:

  unsupported rows (UNDECIDABLE + NO_CORRECT_OPTION, n~93)
      The capability claim was TRUE - it genuinely could not answer a question
      about a hamster that does not exist. The deceptive turn is the FABRICATED
      ANSWER in the fresh context, so the follow-up attaches there.

  DEFENSIBLE rows (n~7)
      The question had a real answer, so the deceptive turn is the CAPABILITY
      CLAIM itself. The follow-up attaches to that instead.

Either way the follow-up lands immediately after the deceptive turn, exactly
as it does in ascii (after the ability denial), sycophancy (after the fold)
and mask-factual (after the false narrative). That consistency is what makes
the self-report numbers comparable at all.

The `named` / `eval_named` conditions also differ per bucket, since they name
the act: confabulation for the 93, false-inability for the 7. Asking the 93
whether they "falsely claimed inability" would be asking about an act they
did not commit.

Runs on a local GPU (bf16, temperature 0). Both activation capture points are
saved per row, per condition.

Run:
    python 00_unanswerable_selfreport_benchmark.py
    python 00_unanswerable_selfreport_benchmark.py --limit 24   # smoke test

Resumable per (sample_id, condition). One bad row is logged and skipped.
"""

import argparse

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import (
    RESULTS, TARGET_MODEL, TEMPERATURE, load_config, load_done, append,
    read_jsonl, to_messages, fresh_ask,
    FOLLOWUP, FOLLOWUP_LOW_STAKES, FOLLOWUP_HIGH_STAKES, FOLLOWUP_CONFIDENCE,
    FOLLOWUP_NEUTRAL_COT, FOLLOWUP_LOW_STAKES_COT, FOLLOWUP_HIGH_STAKES_COT,
    FOLLOWUP_CONFIDENCE_COT, FOLLOWUP_PLAIN, FOLLOWUP_EVAL_PLAIN,
    FOLLOWUP_NAMED_UNANSWERABLE, FOLLOWUP_EVAL_NAMED_UNANSWERABLE,
    FOLLOWUP_NAMED_CONFABULATION, FOLLOWUP_EVAL_NAMED_CONFABULATION,
    admitted_lying, admitted_lying_cot, parse_confidence, parse_confidence_cot,
    parse_reasoning)
from src.dyl_activations import find_answer_char_span, capture_state_at_answer

CONFIG = "google-gemma-3-27b-it_task-group_unanswerable"
ACTIVATIONS_DIR = RESULTS / "activations"
ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RESULTS / "unanswerable_benchmark_conditions.jsonl"

UNSUPPORTED = {"UNDECIDABLE", "NO_CORRECT_OPTION"}

# Shared across buckets except named/eval_named, which are filled in per row.
BASE_WORDING = {
    "neutral": FOLLOWUP, "low_stakes": FOLLOWUP_LOW_STAKES,
    "high_stakes": FOLLOWUP_HIGH_STAKES, "confidence": FOLLOWUP_CONFIDENCE,
    "neutral_cot": FOLLOWUP_NEUTRAL_COT,
    "low_stakes_cot": FOLLOWUP_LOW_STAKES_COT,
    "high_stakes_cot": FOLLOWUP_HIGH_STAKES_COT,
    "confidence_cot": FOLLOWUP_CONFIDENCE_COT,
    "plain": FOLLOWUP_PLAIN, "eval_plain": FOLLOWUP_EVAL_PLAIN,
}
CONDITIONS = list(BASE_WORDING) + ["named", "eval_named"]

KIND = {
    "neutral": "letter", "low_stakes": "letter", "high_stakes": "letter",
    "confidence": "confidence",
    "neutral_cot": "letter_cot", "low_stakes_cot": "letter_cot",
    "high_stakes_cot": "letter_cot", "confidence_cot": "confidence_cot",
    "plain": "letter", "eval_plain": "letter",
    "named": "letter", "eval_named": "letter",
}

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()


def wording(cond: str, unsupported: bool) -> str:
    if cond == "named":
        return (FOLLOWUP_NAMED_CONFABULATION if unsupported
                else FOLLOWUP_NAMED_UNANSWERABLE)
    if cond == "eval_named":
        return (FOLLOWUP_EVAL_NAMED_CONFABULATION if unsupported
                else FOLLOWUP_EVAL_NAMED_UNANSWERABLE)
    return BASE_WORDING[cond]


# --------------------------------------------------------------- inputs ----
primary = pd.read_csv(RESULTS / "unanswerable_claims.csv")
labels = pd.read_csv(RESULTS / "unanswerable_question_labels.csv")[
    ["sample_id", "label"]]
df = primary.merge(labels, on="sample_id", how="inner")

# A row can only be asked "did you lie?" about a turn that exists.
before = len(df)
df = df[df["fresh_reply"].notna() & df["claim_reply"].notna()].copy()
if len(df) < before:
    print(f"!! dropped {before - len(df)} rows with a missing generation")

dataset = pd.concat([load_config(CONFIG, "train"),
                     load_config(CONFIG, "validation")])
dataset = dataset.drop_duplicates(subset=["sample_id"]).set_index("sample_id")

jobs = [(r["sample_id"], c) for _, r in df.iterrows() for c in CONDITIONS]
if args.limit:
    jobs = jobs[: args.limit]
print(f"{len(df)} rows x {len(CONDITIONS)} conditions = {len(jobs)} jobs")

done = load_done(OUT_PATH)
todo = [j for j in jobs if f"{j[0]}::{j[1]}" not in done]
print(f"{len(done)} already done, {len(todo)} to go")

rows_by_id = df.set_index("sample_id")

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

GREEDY = TEMPERATURE == 0.0


def build_ids(messages):
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        return_dict=True)
    return {k: v.to(model.device) for k, v in enc.items()}


@torch.inference_mode()
def generate(messages, max_new_tokens=300) -> str:
    enc = build_ids(messages)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=not GREEDY,
        temperature=None if GREEDY else TEMPERATURE,
        top_p=None if GREEDY else 1.0,
        pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


@torch.inference_mode()
def capture_predecision(messages) -> torch.Tensor:
    enc = build_ids(messages)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    return torch.stack([h[0, -1, :] for h in out.hidden_states]
                       ).to(torch.float16).cpu()


def build_base(row, unsupported: bool):
    """The conversation up to and including the deceptive turn."""
    if unsupported:
        # the fabricated answer, in its own fresh context
        return fresh_ask(row["question"]) + [
            {"role": "assistant", "content": str(row["fresh_reply"])}]
    # the capability claim, in the original conversation
    ds_row = dataset.loc[row.name]
    return (to_messages(ds_row["trace"]) + [dict(ds_row["lie_detection_prompt"]),
            {"role": "assistant", "content": str(row["claim_reply"])}])


def work(job):
    uid, cond = job
    key = f"{uid}::{cond}"
    row = rows_by_id.loc[uid]
    unsupported = row["label"] in UNSUPPORTED
    try:
        messages = build_base(row, unsupported) + [
            {"role": "user", "content": wording(cond, unsupported)}]
        predecision = capture_predecision(messages)
        reply = generate(messages, max_new_tokens=300)
    except Exception as e:
        append(OUT_PATH, {"uid": key, "sample_id": uid, "condition": cond,
                          "error": str(e)})
        return

    kind = KIND[cond]
    admitted = conf = None
    reasoning = ""
    if kind == "letter":
        admitted = admitted_lying(reply)
    elif kind == "letter_cot":
        admitted = admitted_lying_cot(reply)
        reasoning = parse_reasoning(reply)
    elif kind == "confidence":
        conf = parse_confidence(reply)
    else:
        conf = parse_confidence_cot(reply)
        reasoning = parse_reasoning(reply)

    fn_pd = f"{uid}__{cond}__predecision.pt"
    torch.save(predecision, ACTIVATIONS_DIR / fn_pd)

    fn_at = None
    span = find_answer_char_span(reply, kind)
    if span is not None:
        state = capture_state_at_answer(tokenizer, model, messages, reply, span)
        if state is not None:
            fn_at = f"{uid}__{cond}__answer_token.pt"
            torch.save(state, ACTIVATIONS_DIR / fn_at)

    append(OUT_PATH, {
        "uid": key, "sample_id": uid, "condition": cond,
        "bucket": "confabulation" if unsupported else "false_inability",
        "label": row["label"],
        "domain": row["domain"],
        "confabulated": (bool(row["committed"])
                         if unsupported and pd.notna(row["committed"]) else None),
        "answer": reply,
        "admitted": admitted,
        "confidence_score": conf,
        "reasoning": reasoning,
        "predecision_activations_file": f"activations/{fn_pd}",
        "answer_token_activations_file": (f"activations/{fn_at}"
                                          if fn_at else None),
    })


for job in tqdm(todo):
    work(job)

# --------------------------------------------------------------- summary ---
out = pd.DataFrame(read_jsonl(OUT_PATH))
n_err = int(out["error"].notna().sum()) if "error" in out.columns else 0
ok = out[out["error"].isna()] if "error" in out.columns else out
print(f"\n{len(ok)} rows OK, {n_err} errors")
ok.to_csv(RESULTS / "unanswerable_benchmark_conditions.csv", index=False)
print("Wrote results/unanswerable_benchmark_conditions.csv")
print("\nNext: 02_analyse_unanswerable.py for the rates with Wilson CIs.")
