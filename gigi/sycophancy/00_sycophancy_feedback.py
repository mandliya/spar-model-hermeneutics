"""SYCOPHANCY — track B: evaluation bias, with activations.

Does gemma's critique of an argument get nicer or harsher based on a social
cue about who supposedly wrote it or how they feel about it — holding the
argument's actual content fixed? No fact is in dispute here, unlike track A;
the question is purely whether sentiment shifts.

lie_detection_prompt IS the framed ask, for every row — nothing to build.
The dataset itself reuses the same argument across 2-4 framings for most
rows (124 distinct arguments, 110 with a user_did_not_write baseline
available) — a genuine paired design, not something we construct.

For each user_did_not_write row, the SAME framed prompt is also sent a
second time, fresh — a noise floor for how much a critique's tone varies
from sampling alone, with zero social pressure involved.

Every critique gets a 1-10 favorability score from a judge model
(sentiment_score() in src/judge.py — a separate job from the lie judge,
scoring tone, not truthfulness). sentiment_shift is that score minus the
same argument's own did_not_write baseline score (the first of the two
neutral runs, so the second stays free for the noise-floor comparison).

Run:
    python 00_sycophancy_feedback.py
    python 00_sycophancy_feedback.py --limit 12   # smoke test

Cost: ~450 generate calls (local GPU, no per-token cost) + ~450 judge
sentiment calls over OpenRouter (short prompts, cheap).
"""

import argparse

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import (RESULTS, TARGET_MODEL, TEMPERATURE, append, load_done,
                        read_jsonl, load_config, to_messages,
                        extract_argument, argument_group_id)
from src.judge import sentiment_score

CONFIG = "google-gemma-3-27b-it_task-group_sycophancy"
ACTIVATIONS_DIR = RESULTS / "activations"
ACTIVATIONS_DIR.mkdir(exist_ok=True)
OUT_PATH = RESULTS / "sycophancy_feedback_conditions.jsonl"

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=None,
               help="cap the number of jobs — for a smoke test")
args = p.parse_args()

# --------------------------------------------------------------- inputs ----
df = pd.concat([load_config(CONFIG, "train"), load_config(CONFIG, "validation")])
df = df[df["task"].str.startswith("sycophancy_feedback")].copy()
print(f"{len(df)} feedback rows across {df['task'].value_counts().to_dict()}")

df["argument"] = df["lie_detection_prompt"].apply(
    lambda p: extract_argument(p["content"]))
n_bad = df["argument"].isna().sum()
if n_bad:
    print(f"!! {n_bad} rows failed to parse — dropping, not guessing")
df = df[df["argument"].notna()].copy()
df["gid"] = df["argument"].apply(argument_group_id)
print(f"{len(df)} rows across {df['gid'].nunique()} distinct arguments")

# jobs: every row once, plus a second independent run for each
# user_did_not_write row (the noise floor). Interleaved (main immediately
# followed by its noise_rerun, for did_not_write rows) rather than all mains
# then all reruns, so a small --limit smoke test actually exercises both
# paths instead of only ever reaching "main" jobs.
jobs = []
for uid, task in zip(df["sample_id"], df["task"]):
    jobs.append((uid, "main"))
    if task == "sycophancy_feedback_user_did_not_write":
        jobs.append((uid, "noise_rerun"))
if args.limit:
    jobs = jobs[: args.limit]
print(f"{len(jobs)} jobs to run")

df = df.set_index("sample_id")

# --------------------------------------------------------------- model -----
print(f"Loading {TARGET_MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)
model.eval()
print("Model loaded.")


def build_ids(messages: list[dict]):
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        return_dict=True,
    )
    return {k: v.to(model.device) for k, v in enc.items()}


@torch.inference_mode()
def generate(messages: list[dict], max_new_tokens: int = 200) -> str:
    enc = build_ids(messages)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=TEMPERATURE > 0,
        temperature=TEMPERATURE if TEMPERATURE > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@torch.inference_mode()
def capture_state(messages: list[dict]) -> torch.Tensor:
    """Same pre-decision capture as every other condition: every layer's
    hidden state at the last prompt token, clean no-cache forward pass."""
    enc = build_ids(messages)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    last_token = torch.stack([h[0, -1, :] for h in out.hidden_states])
    return last_token.to(torch.float16).cpu()


# ------------------------------------------------------------------ main ---
done = load_done(OUT_PATH)
todo = [(uid, kind) for uid, kind in jobs if f"{uid}::{kind}" not in done]
print(f"{len(done)} already done, {len(todo)} to go")


def work(job):
    uid, kind = job
    row = df.loc[uid]
    msgs = to_messages(row["trace"])
    probe = dict(row["lie_detection_prompt"])
    convo = msgs + [probe]

    try:
        state = capture_state(convo)
        critique = generate(convo)
        score, judge_raw = sentiment_score(row["argument"], critique)
    except Exception as e:
        append(OUT_PATH, {"uid": f"{uid}::{kind}", "error": str(e)})
        return

    activations_file = f"activations/{uid}__{kind}.pt"
    torch.save(state, ACTIVATIONS_DIR / f"{uid}__{kind}.pt")

    append(OUT_PATH, {
        "uid": f"{uid}::{kind}",
        "sample_id": uid,
        "kind": kind,                 # "main" or "noise_rerun"
        "task": row["task"],
        "argument_group_id": row["gid"],
        "critique": critique,
        "sentiment_score": score,
        "judge_raw": judge_raw,
        "activations_file": activations_file,
    })


# No ThreadPoolExecutor — one shared GPU, single CUDA context, sequential.
for job in tqdm(todo):
    work(job)

# --------------------------------------------------------------- results ---
res = pd.DataFrame([r for r in read_jsonl(OUT_PATH) if "error" not in r])
main = res[res["kind"] == "main"].copy()

# baseline per argument group: that group's own did_not_write "main" score
baselines = (main[main["task"] == "sycophancy_feedback_user_did_not_write"]
             .set_index("argument_group_id")["sentiment_score"])
main["baseline"] = main["argument_group_id"].map(baselines)
main["sentiment_shift"] = main["sentiment_score"] - main["baseline"]

print("\n=== mean sentiment score by condition ===")
for task in sorted(main["task"].unique()):
    sub = main[main["task"] == task].dropna(subset=["sentiment_score"])
    print(f"  {task:40s} mean={sub['sentiment_score'].mean():.2f}  n={len(sub)}")

print("\n=== mean sentiment SHIFT vs that argument's own neutral baseline ===")
for task in ["sycophancy_feedback_user_wrote", "sycophancy_feedback_user_likes",
             "sycophancy_feedback_user_dislikes"]:
    sub = main[main["task"] == task].dropna(subset=["sentiment_shift"])
    if len(sub):
        print(f"  {task:40s} mean shift={sub['sentiment_shift'].mean():+.2f}  n={len(sub)}")

noise = res[res["kind"] == "noise_rerun"].copy()
noise_pairs = noise.merge(
    main[main["task"] == "sycophancy_feedback_user_did_not_write"],
    on="argument_group_id", suffixes=("_2", "_1"))
noise_pairs["gap"] = (noise_pairs["sentiment_score_2"]
                      - noise_pairs["sentiment_score_1"]).abs()
print(f"\n=== noise floor: |score gap| between two neutral reruns of the "
      f"same argument ===")
print(f"  mean |gap| = {noise_pairs['gap'].mean():.2f}  n={len(noise_pairs)}")

res.to_csv(RESULTS / "sycophancy_feedback_conditions.csv", index=False)
print(f"\nFull table: {RESULTS / 'sycophancy_feedback_conditions.csv'}")
