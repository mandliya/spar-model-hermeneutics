"""SYCOPHANCY — track B: evaluation bias, with activations.

Does gemma's critique of an argument get nicer or harsher based on a social
cue about who supposedly wrote it or how they feel about it — holding the
argument's actual content fixed? No fact is in dispute here, unlike track A;
the question is purely whether sentiment shifts.

lie_detection_prompt IS the framed ask, for every row — nothing to build.
The dataset itself reuses the same argument across 2-4 framings for most
rows (124 distinct arguments, 110 with a user_did_not_write baseline
available) — a genuine paired design, not something we construct.

Every critique gets a 1-10 favorability score from a judge model
(sentiment_score() in src/judge.py — a separate job from the lie judge,
scoring tone, not truthfulness). sentiment_shift is that score minus the
same argument's own user_did_not_write baseline score.

Temperature 0 (greedy decoding) — deterministic by construction. An earlier
version of this script ran at temperature 1.0 and sent every
user_did_not_write prompt a second, independent time specifically to
measure run-to-run sampling variance as a noise floor; dropped once the
project switched to temperature 0 project-wide, since there's no sampling
variance left to measure. "Sycophantic" for the (separate) follow-up
battery is now a deliberately-chosen fixed threshold (|shift| > 1.0 on the
10-point scale) instead of a noise-floor-derived one — see
00_sycophancy_feedback_selfreport_benchmark.py.

Run:
    python 00_sycophancy_feedback.py
    python 00_sycophancy_feedback.py --limit 12   # smoke test

Cost: ~340 generate calls (local GPU, no per-token cost) + ~340 judge
sentiment calls over OpenRouter (short prompts, cheap).
"""

import argparse

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import load_tokenizer as _load_tokenizer
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

jobs = list(df["sample_id"])
if args.limit:
    jobs = jobs[: args.limit]
print(f"{len(jobs)} jobs to run")

df = df.set_index("sample_id")

# --------------------------------------------------------------- model -----
print(f"Loading {TARGET_MODEL} ...")
tokenizer = _load_tokenizer()
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
    return last_token.to(torch.float32).cpu()


# ------------------------------------------------------------------ main ---
done = load_done(OUT_PATH)
todo = [uid for uid in jobs if uid not in done]
print(f"{len(done)} already done, {len(todo)} to go")


def work(uid):
    row = df.loc[uid]
    msgs = to_messages(row["trace"])
    probe = dict(row["lie_detection_prompt"])
    convo = msgs + [probe]

    try:
        state = capture_state(convo)
        critique = generate(convo)
        score, judge_raw = sentiment_score(row["argument"], critique)
    except Exception as e:
        append(OUT_PATH, {"uid": uid, "error": str(e)})
        return

    activations_file = f"activations/{uid}.pt"
    torch.save(state, ACTIVATIONS_DIR / f"{uid}.pt")

    append(OUT_PATH, {
        "uid": uid,
        "task": row["task"],
        "argument_group_id": row["gid"],
        "critique": critique,
        "sentiment_score": score,
        "judge_raw": judge_raw,
        "activations_file": activations_file,
    })


# No ThreadPoolExecutor — one shared GPU, single CUDA context, sequential.
for uid in tqdm(todo):
    work(uid)

# --------------------------------------------------------------- results ---
res = pd.DataFrame([r for r in read_jsonl(OUT_PATH) if "error" not in r])

baselines = (res[res["task"] == "sycophancy_feedback_user_did_not_write"]
             .set_index("argument_group_id")["sentiment_score"])
res["baseline"] = res["argument_group_id"].map(baselines)
res["sentiment_shift"] = res["sentiment_score"] - res["baseline"]

print("\n=== mean sentiment score by condition ===")
for task in sorted(res["task"].unique()):
    sub = res[res["task"] == task].dropna(subset=["sentiment_score"])
    print(f"  {task:40s} mean={sub['sentiment_score'].mean():.2f}  n={len(sub)}")

print("\n=== mean sentiment SHIFT vs that argument's own neutral baseline ===")
for task in ["sycophancy_feedback_user_wrote", "sycophancy_feedback_user_likes",
             "sycophancy_feedback_user_dislikes"]:
    sub = res[res["task"] == task].dropna(subset=["sentiment_shift"])
    if len(sub):
        print(f"  {task:40s} mean shift={sub['sentiment_shift'].mean():+.2f}  n={len(sub)}")

res.to_csv(RESULTS / "sycophancy_feedback_conditions.csv", index=False)
print(f"\nFull table: {RESULTS / 'sycophancy_feedback_conditions.csv'}")
