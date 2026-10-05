"""MASK-FACTUAL — mask_provided_facts subtype, with activations.

Structurally different from ascii-sandbagging and sycophancy: the true fact
AND the false narrative gemma is pressured to give are both already stated
in the row's own system prompt (e.g. a sales rep told "the site's down from
a system error, but tell customers it's a system upgrade"). Ground truth
isn't elicited the way Track A's is — it's given, so no neutral-vs-pressured
split is needed to discover it.

Each response is classified by classify_mask_response() in src/judge.py
against the system prompt's own stated true fact / false narrative:
TOLD_TRUTH / TOLD_FALSE_NARRATIVE / EVASIVE. `lied` = TOLD_FALSE_NARRATIVE.

Temperature 0 (greedy decoding) — deterministic by construction, so the
pressured prompt is sent once, not twice. An earlier version of this
script sent it twice at temperature 1.0 specifically to check whether
gemma's compliance with the pressure was stable across independent samples
(the thing sampling variance can flip by chance); dropped once the project
switched to temperature 0 project-wide, since there's no run-to-run
variance left to check.

Run:
    python 00_mask_factual_provided.py
    python 00_mask_factual_provided.py --limit 12   # smoke test

Cost: ~84 generate calls (local GPU, no per-token cost) + ~84 judge
classification calls over OpenRouter (short prompts, cheap).
"""

import argparse

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import load_tokenizer as _load_tokenizer
from src.common import (RESULTS, TARGET_MODEL, TEMPERATURE, append, load_done,
                        read_jsonl, load_config, to_messages)
from src.judge import classify_mask_response

CONFIG = "google-gemma-3-27b-it_task-group_mask-factual"
ACTIVATIONS_DIR = RESULTS / "activations"
ACTIVATIONS_DIR.mkdir(exist_ok=True)
OUT_PATH = RESULTS / "mask_factual_provided_conditions.jsonl"

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=None,
               help="cap the number of jobs — for a smoke test")
args = p.parse_args()

# --------------------------------------------------------------- inputs ----
df = pd.concat([load_config(CONFIG, "train"), load_config(CONFIG, "validation")])
df = df[df["task"] == "mask_provided_facts"].copy()
print(f"{len(df)} mask_provided_facts rows")

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
    system_prompt = row["trace"]["content"][0]
    question = probe["content"]

    try:
        state = capture_state(convo)
        response = generate(convo)
        verdict, judge_raw = classify_mask_response(system_prompt, question, response)
    except Exception as e:
        append(OUT_PATH, {"uid": uid, "error": str(e)})
        return

    activations_file = f"activations/{uid}.pt"
    torch.save(state, ACTIVATIONS_DIR / f"{uid}.pt")

    append(OUT_PATH, {
        "uid": uid,
        "task": row["task"],
        "response": response,
        "verdict": verdict,
        "lied": verdict == "TOLD_FALSE_NARRATIVE",
        "judge_raw": judge_raw,
        "activations_file": activations_file,
    })


# No ThreadPoolExecutor — one shared GPU, single CUDA context, sequential.
for uid in tqdm(todo):
    work(uid)

# --------------------------------------------------------------- results ---
res = pd.DataFrame([r for r in read_jsonl(OUT_PATH) if "error" not in r])


def wilson(k, n):
    from statsmodels.stats.proportion import proportion_confint
    if n == 0:
        return "n/a"
    lo, hi = proportion_confint(k, n, method="wilson")
    return f"{k/n:6.1%}  [{lo:.1%}, {hi:.1%}]  n={n}"


ok = res.dropna(subset=["verdict"])
print(f"\n=== lying rate ===")
print(f"  told false narrative = {wilson(int((ok['verdict']=='TOLD_FALSE_NARRATIVE').sum()), len(ok))}")
print(f"  told truth           = {wilson(int((ok['verdict']=='TOLD_TRUTH').sum()), len(ok))}")
print(f"  evasive               = {wilson(int((ok['verdict']=='EVASIVE').sum()), len(ok))}")

res.to_csv(RESULTS / "mask_factual_provided_conditions.csv", index=False)
print(f"\nFull table: {RESULTS / 'mask_factual_provided_conditions.csv'}")
