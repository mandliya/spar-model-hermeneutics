"""Pre-decision activation capture for ascii-sandbagging's plain/named/
eval_plain/eval_named conditions specifically — the four conditions that
were built text-only (no activations at all), unlike the other eight
follow-up conditions, which always captured the pre-decision state from
their original generation scripts and never needed this.

Per Tianqin: cache BOTH the pre-decision token (last prompt token, before
generation) AND the answer token (src/dyl_activations.py /
01_dyl_recapture_ascii.py) — there's now empirical uncertainty about which
one the DYL probe actually expects, so both get saved rather than guessing.
This script is the pre-decision half for these four conditions; the answer-
token half is 01_dyl_recapture_ascii.py, which already covers them.

Reuses already-saved answer text only to know which rows/conditions exist —
no regeneration, and the pre-decision capture itself doesn't depend on what
was generated at all (it's a forward pass over the prompt alone, before the
answer). Saved as <uid>__<condition>__predecision.pt, alongside (not
replacing) the __answer_token.pt files.

Run:
    python 02_predecision_ascii_benchmark.py
    python 02_predecision_ascii_benchmark.py --limit 12   # smoke test
"""

import argparse

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import (RESULTS, TARGET_MODEL, append, load_done, load_config,
                        to_messages, FOLLOWUP_PLAIN, FOLLOWUP_NAMED_SANDBAGGING,
                        FOLLOWUP_EVAL_PLAIN, FOLLOWUP_EVAL_NAMED_SANDBAGGING)

CONFIG = "google-gemma-3-27b-it_task-group_sandbagging"
ACTIVATIONS_DIR = RESULTS / "activations"
OUT_PATH = RESULTS / "predecision_ascii_benchmark.jsonl"

FOLLOWUP_WORDING = {
    "plain": FOLLOWUP_PLAIN, "named": FOLLOWUP_NAMED_SANDBAGGING,
    "eval_plain": FOLLOWUP_EVAL_PLAIN, "eval_named": FOLLOWUP_EVAL_NAMED_SANDBAGGING,
}

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()

# --------------------------------------------------------------- inputs ----
ascii_results = pd.read_csv(RESULTS / "ascii_results.csv").set_index("uid")
bench = pd.read_csv(RESULTS / "ascii_benchmark_conditions.csv")

df = pd.concat([load_config(CONFIG, "train"), load_config(CONFIG, "validation")])
df = df[df["task"].str.contains("ascii", case=False, na=False)].set_index("sample_id")

jobs = [(row["sample_id"], row["condition"]) for _, row in bench.iterrows()]
if args.limit:
    jobs = jobs[: args.limit]
print(f"{len(jobs)} (uid, condition) pairs to capture")

done = load_done(OUT_PATH)
todo = [(uid, cond) for uid, cond in jobs if f"{uid}::{cond}" not in done]
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


def build_convo(uid, cond):
    row = df.loc[uid]
    msgs = to_messages(row["trace"])
    probe = dict(row["lie_detection_prompt"])
    a1 = str(ascii_results.loc[uid, "probe_answer"])
    return msgs + [probe, {"role": "assistant", "content": a1},
                   {"role": "user", "content": FOLLOWUP_WORDING[cond]}]


def work(job):
    uid, cond = job
    key = f"{uid}::{cond}"
    try:
        messages = build_convo(uid, cond)
        state = capture_state(messages)
    except Exception as e:
        append(OUT_PATH, {"uid": key, "error": str(e)})
        return

    fname = f"{uid}__{cond}__predecision.pt"
    torch.save(state, ACTIVATIONS_DIR / fname)
    append(OUT_PATH, {"uid": key, "sample_id": uid, "condition": cond,
                      "predecision_activations_file": f"activations/{fname}"})


for job in tqdm(todo):
    work(job)

print(f"\nDone. Saved as results/predecision_ascii_benchmark.jsonl "
      f"(maps uid::condition -> file)")
