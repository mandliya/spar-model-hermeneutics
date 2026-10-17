"""Does batching change gemma's answers or activations?

Batching stacks several prompts into one forward pass. An attention mask stops
any row from attending to another, so in exact arithmetic each answer is
unchanged. But the model runs in bf16, and a different batch shape changes the
order the GPU sums numbers in - which can change the last digits, flip a
near-tied greedy choice, and send the rest of an answer down a different path.

This measures both effects against data we already trust:

  answers      regenerate saved follow-up answers at batch size 1 (control -
               already shown to reproduce 50/50), 8 and 16, and count exact
               matches against what's on disk.
  activations  capture the pre-decision hidden state for the same rows at
               batch 1 and batch 16, and report how far apart they are.

Two categories, deliberately different: ascii (short prompts) and
capture-the-secret (long role-play prompts), because mixed lengths mean more
left-padding, which is where batching bugs surface.

Forward passes and generation only. No API calls. Writes nothing to results/
except results/batching_check.json, a new file.

Run:
    python 99_batching_check.py --n 48
"""

import argparse
import json

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import (RESULTS, TARGET_MODEL, TEMPERATURE, load_config,
                        to_messages, FOLLOWUP)

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=48, help="rows per category")
args = p.parse_args()

if TEMPERATURE != 0.0:
    raise SystemExit("!! TEMPERATURE must be 0.0 for this check to mean anything")

# ---------------------------------------------------------------- rows -----
def ascii_rows(n):
    saved = pd.read_csv(RESULTS / "ascii_results.csv")
    saved = saved[saved.followup_answer.notna() & saved.probe_answer.notna()]
    ds = pd.concat([load_config("google-gemma-3-27b-it_task-group_sandbagging", s)
                    for s in ["train", "validation"]])
    ds = ds[ds.task.str.contains("ascii", case=False, na=False)]
    ds = ds.drop_duplicates(subset=["sample_id"]).set_index("sample_id")
    out = []
    for _, r in saved[saved.uid.isin(ds.index)].head(n).iterrows():
        d = ds.loc[r.uid]
        msgs = to_messages(d.trace) + [
            dict(d.lie_detection_prompt),
            {"role": "assistant", "content": str(r.probe_answer)},
            {"role": "user", "content": FOLLOWUP}]
        out.append(("ascii", r.uid, msgs, str(r.followup_answer).strip(), 120))
    return out


def capture_rows(n):
    saved = pd.read_csv(RESULTS / "capture_secret_benchmark_conditions.csv",
                        low_memory=False)
    saved = saved[(saved.condition == "neutral") & saved.answer.notna()]
    prim = pd.read_csv(RESULTS / "capture_secret_conditions.csv").set_index("sample_id")
    ds = pd.concat([load_config("google-gemma-3-27b-it_task-group_games", s)
                    for s in ["train", "validation"]])
    ds = ds.drop_duplicates(subset=["sample_id"]).set_index("sample_id")
    out = []
    for _, r in saved.drop_duplicates(subset=["sample_id"]).head(n).iterrows():
        if r.sample_id not in ds.index or r.sample_id not in prim.index:
            continue
        d = ds.loc[r.sample_id]
        msgs = to_messages(d.trace) + [
            dict(d.lie_detection_prompt),
            {"role": "assistant", "content": str(prim.loc[r.sample_id, "reply"])},
            {"role": "user", "content": FOLLOWUP}]
        out.append(("capture", r.sample_id, msgs, str(r.answer).strip(), 80))
    return out


rows = ascii_rows(args.n) + capture_rows(args.n)
print(f"{len(rows)} rows  (ascii + capture-the-secret)")

# --------------------------------------------------------------- model -----
print(f"Loading {TARGET_MODEL} ...")
tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
tok.padding_side = "left"          # generation needs left padding
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager")
model.eval()
print("Model loaded.")


def encode(batch_msgs):
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in batch_msgs]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return {k: v.to(model.device) for k, v in enc.items()}


@torch.inference_mode()
def generate_batch(batch_msgs, max_new):
    enc = encode(batch_msgs)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         temperature=None, top_p=None,
                         pad_token_id=tok.pad_token_id)
    L = enc["input_ids"].shape[1]
    return [tok.decode(o[L:], skip_special_tokens=True).strip() for o in out]


@torch.inference_mode()
def predecision_batch(batch_msgs):
    """Last-prompt-token hidden state, all layers. With left padding the last
    position is the real last token for every row."""
    enc = encode(batch_msgs)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = torch.stack(out.hidden_states)          # [layers, batch, seq, dim]
    return hs[:, :, -1, :].float().cpu()         # [layers, batch, dim]


# -------------------------------------------------------- answer check -----
results = {}
for bs in [1, 8, 16]:
    exact = {"ascii": [0, 0], "capture": [0, 0]}
    mismatches = []
    for i in tqdm(range(0, len(rows), bs), desc=f"generate bs={bs}"):
        chunk = rows[i:i + bs]
        # generate each category's chunk with its own token budget, matching
        # how the original answers were produced
        for cat in ("ascii", "capture"):
            sub = [r for r in chunk if r[0] == cat]
            if not sub:
                continue
            outs = generate_batch([r[2] for r in sub], sub[0][4])
            for r, got in zip(sub, outs):
                exact[cat][1] += 1
                if got == r[3]:
                    exact[cat][0] += 1
                elif len(mismatches) < 6:
                    mismatches.append({"cat": cat, "id": r[1],
                                       "saved": r[3][:100], "batched": got[:100]})
    results[f"answers_bs{bs}"] = {"exact": exact, "mismatches": mismatches}
    for cat, (k, n) in exact.items():
        print(f"  bs={bs:2d}  {cat:8s} exact {k}/{n}  ({100*k/max(n,1):.1f}%)")

# ---------------------------------------------------- activation check -----
print("\ncapturing pre-decision activations at bs=1 and bs=16 ...")
msgs_all = [r[2] for r in rows]
ref = torch.cat([predecision_batch([m]) for m in tqdm(msgs_all, desc="bs=1")], dim=1)
bat = torch.cat([predecision_batch(msgs_all[i:i + 16])
                 for i in tqdm(range(0, len(msgs_all), 16), desc="bs=16")], dim=1)

diff = (ref - bat).abs()
cos = torch.nn.functional.cosine_similarity(ref, bat, dim=-1)   # [layers, rows]
rel = diff.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)
act = {
    "max_abs_diff": float(diff.max()),
    "mean_rel_diff_last_layer": float(rel[-1].mean()),
    "mean_rel_diff_all_layers": float(rel.mean()),
    "min_cosine_any_layer_any_row": float(cos.min()),
    "mean_cosine_last_layer": float(cos[-1].mean()),
}
results["activations_bs1_vs_bs16"] = act
print("\nactivations, bs=1 vs bs=16:")
for k, v in act.items():
    print(f"  {k:32s} {v:.6g}")

out = RESULTS / "batching_check.json"
out.write_text(json.dumps(results, indent=2))
print(f"\nwrote {out}")

ok = all(results[f"answers_bs{b}"]["exact"][c][0] == results[f"answers_bs{b}"]["exact"][c][1]
         for b in (8, 16) for c in ("ascii", "capture"))
print("\nVERDICT:", "batched answers IDENTICAL to saved answers" if ok
      else "batched answers DIFFER from saved answers on some rows — see mismatches")
