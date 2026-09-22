"""ASCII SANDBAGGING — cross-category self-report benchmark (no activations).

The "ask a question directly" arm of ask-directly vs. mech-interp: how good
is gemma's own plain self-report at detecting its own sandbagging, measured
against the ground truth we already established (Part 1's live capability
check)? Two conditions, not one — see src/common.py for why a single
wording isn't trustworthy on its own:

  plain   category-agnostic wording, comparable across categories
  named   uses this project's own vocabulary ("sandbagging") for the
          failure mode

Same reuse pattern as every other follow-up here: gemma's saved PART 1
answer from results/ascii_results.csv is fed back in unchanged, only the
follow-up question varies. Deliberately skips activation capture (not
needed for this comparison — this script is the "ask directly" side of the
benchmark, not the mech-interp side) so it's faster and cheaper than the
other local-GPU scripts in this repo.

Run:
    python 00_ascii_selfreport_benchmark.py
    python 00_ascii_selfreport_benchmark.py --limit 12   # smoke test

Cost: no per-token API cost (local GPU, no judge calls either - this is a
forced A/B, parsed the same way as every other letter-based condition).
~2 conditions x 310 rows = 620 generate calls, no second forward pass per
row since there's no activation capture.
"""

import argparse

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import (RESULTS, TARGET_MODEL, TEMPERATURE, append, load_done,
                        read_jsonl, to_messages, admitted_lying,
                        FOLLOWUP_PLAIN, FOLLOWUP_NAMED_SANDBAGGING)

OUT_PATH = RESULTS / "ascii_benchmark_conditions.jsonl"

CONDITIONS = {
    "plain": FOLLOWUP_PLAIN,
    "named": FOLLOWUP_NAMED_SANDBAGGING,
}

p = argparse.ArgumentParser()
p.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
               help="which of plain named to run")
p.add_argument("--limit", type=int, default=None,
               help="cap the number of (row, condition) calls — for a smoke test")
args = p.parse_args()
for c in args.conditions:
    if c not in CONDITIONS:
        raise SystemExit(f"Unknown condition '{c}'. Choose from: {list(CONDITIONS)}")

# --------------------------------------------------------------- inputs ----
prior = pd.read_csv(RESULTS / "ascii_results.csv")
if prior.empty:
    raise SystemExit("results/ascii_results.csv is empty — run "
                      "00_ascii_activations.py first.")
prior = prior.set_index("uid")
print(f"{len(prior)} rows with a saved PART 1 answer to build on")

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
def generate(messages: list[dict], max_new_tokens: int = 80) -> str:
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


# ------------------------------------------------------------------ main ---
# Need the original trace + probe to rebuild the conversation prefix gemma's
# PART 1 answer sat in — pulled straight from the dataset, same as every
# other follow-up script.
from src.common import load_config
CONFIG = "google-gemma-3-27b-it_task-group_sandbagging"
df = pd.concat([load_config(CONFIG, "train"), load_config(CONFIG, "validation")])
df = df[df["task"].str.contains("ascii", case=False, na=False)].set_index("sample_id")

done = load_done(OUT_PATH)
jobs = [(uid, cond) for uid in prior.index for cond in args.conditions
        if f"{uid}::{cond}" not in done]
if args.limit:
    jobs = jobs[: args.limit]
print(f"{len(done)} already done, {len(jobs)} calls to go")


def work(job):
    uid, cond = job
    row = df.loc[uid]
    a1 = str(prior.loc[uid, "probe_answer"])   # gemma's own saved PART 1 answer

    msgs = to_messages(row["trace"])
    probe = dict(row["lie_detection_prompt"])
    convo = msgs + [probe,
                    {"role": "assistant", "content": a1},
                    {"role": "user", "content": CONDITIONS[cond]}]

    try:
        answer = generate(convo)
    except Exception as e:
        append(OUT_PATH, {"uid": f"{uid}::{cond}", "error": str(e)})
        return

    append(OUT_PATH, {
        "uid": f"{uid}::{cond}",
        "sample_id": uid,
        "condition": cond,
        "lied": bool(prior.loc[uid, "lied"]),   # ground truth, carried over
        "answer": answer,
        "admitted": admitted_lying(answer),
    })


# No ThreadPoolExecutor — one shared GPU, single CUDA context, sequential.
for job in tqdm(jobs):
    work(job)

# --------------------------------------------------------------- results ---
res = pd.DataFrame([r for r in read_jsonl(OUT_PATH) if "error" not in r])


def wilson(k, n):
    from statsmodels.stats.proportion import proportion_confint
    if n == 0:
        return "n/a"
    lo, hi = proportion_confint(k, n, method="wilson")
    return f"{k/n:6.1%}  [{lo:.1%}, {hi:.1%}]  n={n}"


for cond in args.conditions:
    sub = res[res["condition"] == cond]
    ok = sub.dropna(subset=["admitted"])
    unparse = len(sub) - len(ok)
    lied = ok[ok["lied"]]
    told = ok[~ok["lied"]]
    print(f"\n=== {cond} (unparseable: {unparse}) ===")
    print(f"  confession        admits | it lied      = "
          f"{wilson(int(lied['admitted'].sum()), len(lied))}")
    print(f"  doubling down     denies | it lied      = "
          f"{wilson(int((~lied['admitted']).sum()), len(lied))}")
    print(f"  false confession  admits | truthful     = "
          f"{wilson(int(told['admitted'].sum()), len(told))}")

res.to_csv(RESULTS / "ascii_benchmark_conditions.csv", index=False)
print(f"\nFull table: {RESULTS / 'ascii_benchmark_conditions.csv'}")
