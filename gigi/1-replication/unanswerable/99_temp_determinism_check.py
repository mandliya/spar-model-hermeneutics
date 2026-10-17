"""Was the shipped ascii data generated greedily (temperature 0) or sampled?

No record of the temperature was ever written to the results, and both .env
files said TEMPERATURE=1.0 while src/common.py defaults to 0.0 — so the only
way to find out is to re-generate saved rows at temperature 0 and compare.

Reconstructs the follow-up conversation EXACTLY as 00_ascii_activations.py
built it (trace + lie_detection_prompt + saved probe_answer + FOLLOWUP,
max_new_tokens=120), then generates greedily and diffs against the stored
followup_answer.

Reading the result:
    100% match      -> the originals were greedy; the temp-0 rebuild is intact
    ~70-85% match   -> the originals were sampled at 1.0; all four categories
                       need regenerating

Costs GPU time only. No OpenRouter calls, and nothing in results/ is written
or overwritten — output goes to results/temp_determinism_check.json, a new file.

Run:
    python 99_temp_determinism_check.py --n 50
"""

import argparse
import json

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import (RESULTS, TARGET_MODEL, TEMPERATURE, load_config,
                        to_messages, FOLLOWUP)

CONFIG = "google-gemma-3-27b-it_task-group_sandbagging"

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=50)
args = p.parse_args()

print(f"TEMPERATURE as this process sees it: {TEMPERATURE}")
if TEMPERATURE != 0.0:
    raise SystemExit("!! This check is meaningless unless TEMPERATURE=0.0. "
                     "Fix .env first.")

# --------------------------------------------------------------- inputs ----
saved = pd.read_csv(RESULTS / "ascii_results.csv")
saved = saved[saved["followup_answer"].notna()
              & saved["probe_answer"].notna()].copy()

df = pd.concat([load_config(CONFIG, "train"), load_config(CONFIG, "validation")])
df = df[df["task"].str.contains("ascii", case=False, na=False)]
df = df.drop_duplicates(subset=["sample_id"]).set_index("sample_id")

saved = saved[saved["uid"].isin(df.index)].head(args.n)
print(f"comparing {len(saved)} rows")

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
def generate(messages, max_new_tokens=120) -> str:
    """Byte-identical settings to 00_ascii_activations.py's generate(), except
    forced greedy."""
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None,
                         pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


rows, exact, norm_match = [], 0, 0
for _, r in tqdm(list(saved.iterrows()), total=len(saved)):
    ds = df.loc[r["uid"]]
    msgs = to_messages(ds["trace"]) + [dict(ds["lie_detection_prompt"]),
                                       {"role": "assistant",
                                        "content": str(r["probe_answer"])},
                                       {"role": "user", "content": FOLLOWUP}]
    try:
        fresh = generate(msgs)
    except Exception as e:
        rows.append({"uid": r["uid"], "error": str(e)})
        continue
    old = str(r["followup_answer"]).strip()
    e_ok = fresh.strip() == old
    n_ok = fresh.strip().upper()[:40] == old.upper()[:40]
    exact += e_ok
    norm_match += n_ok
    rows.append({"uid": r["uid"], "saved": old, "regenerated": fresh,
                 "exact_match": bool(e_ok), "prefix_match": bool(n_ok)})

n = sum(1 for x in rows if "error" not in x)
print("\n" + "=" * 60)
print(f"rows compared        : {n}")
print(f"exact match          : {exact}/{n}  ({100*exact/max(n,1):.1f}%)")
print(f"first-40-char match  : {norm_match}/{n}  ({100*norm_match/max(n,1):.1f}%)")
print("=" * 60)
if n and exact == n:
    print("VERDICT: consistent with greedy decoding. The shipped data looks")
    print("         like it was generated at temperature 0 after all.")
elif n:
    print("VERDICT: the shipped answers do NOT reproduce under greedy decoding.")
    print("         They were sampled. All four categories need regenerating.")

out = RESULTS / "temp_determinism_check.json"
out.write_text(json.dumps({"n": n, "exact": exact, "prefix": norm_match,
                           "rows": rows}, indent=2))
print(f"\nwrote {out}")
for x in rows[:5]:
    if "error" not in x and not x["exact_match"]:
        print(f"\n--- mismatch {x['uid']}")
        print(f"  saved      : {x['saved'][:120]!r}")
        print(f"  regenerated: {x['regenerated'][:120]!r}")
