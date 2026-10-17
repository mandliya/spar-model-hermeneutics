"""ASCII SANDBAGGING + ACTIVATIONS — local-GPU version of 00_ascii.py.

Runs entirely on a self-hosted GPU (Vast.ai) instead of OpenRouter, because
OpenRouter (and every hosted inference API) only ever returns generated text —
it has no mechanism to hand back internal hidden states. Getting activations
means loading the actual weights ourselves and reading them out of the forward
pass directly.

Text logic (prompts, letter-parsing, what counts as a lie) is IDENTICAL to
00_ascii.py — both import the same helpers from src/common.py, so "lied" and
"admitted" can never quietly drift between the two runs.

What's new here: for each row, in addition to generating gemma's answers, we
do one extra no-cache forward pass at each of two points and save the model's
internal state (every layer, last token of the prompt) at that moment:

  lied_state          — last token right before gemma answers the capability
                         probe. This is the "pre-decision" state: gemma has
                         read the question but said nothing yet.
  doubled_down_state  — last token right before gemma answers our follow-up
                         ("did you lie?"), same idea, one turn later.

We deliberately do NOT harvest these from inside model.generate() — with KV
caching, hidden states arrive incrementally per generated token and you'd
have to reassemble them, which is fragile. Instead: generate normally, throw
the KV cache away, then do one clean extra forward pass over the exact same
prompt with output_hidden_states=True. It's a second prefill of a few hundred
tokens — trivially cheap next to the generation itself — and it gives exactly
reproducible states every time.

Scope, as decided: two capture points only (no harmless-question negative
controls yet), all layers, last prompt token only. See RESULTS.md / chat
history for why — this can be extended later without touching what's here.

Requirements (NOT the same as the rest of this repo — see requirements-vast.txt):
    pip install -r requirements-vast.txt
Needs an HF token with access to google/gemma-3-27b-it (it's gated):
    export HF_TOKEN=hf_...

Run:
    python 00_ascii_activations.py
    python 00_ascii_activations.py --n 620

Cost: no per-token API cost — you're paying for GPU-hours instead. No
parallelism across rows (see note below) — this runs one row at a time.
"""

import argparse
import pathlib

import pandas as pd
import torch
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import (RESULTS, TARGET_MODEL, TEMPERATURE, append, load_done,
                        read_jsonl, load_config, sanity_check_labels,
                        to_messages, FOLLOWUP, CAPABILITY_CHECK,
                        claimed_inability, admitted_lying)

CONFIG = "google-gemma-3-27b-it_task-group_sandbagging"
ACTIVATIONS_DIR = RESULTS / "activations"
ACTIVATIONS_DIR.mkdir(exist_ok=True)
OUT_PATH = RESULTS / "ascii_results.jsonl"

# --------------------------------------------------------------- model -----
# eager attention, not sdpa/flash — an explicit, honest setting for a run
# whose whole point is inspecting internals cleanly, not raw throughput.
# device_map="auto" rather than a hardcoded "cuda" so this still works if the
# box ends up with more than one GPU.
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
    """Tokenize a chat, with the generation prompt appended, ready either to
    feed to .generate() or to run through a plain forward pass.

    return_dict=True explicitly, rather than trusting the default: recent
    transformers versions return a BatchEncoding (dict-like) here, not a bare
    tensor, and passing that straight to model.generate() as a positional arg
    fails opaquely (it tries `.shape` on the dict). Unpacking input_ids +
    attention_mask explicitly is also more correct than a bare tensor would
    be, not just a workaround.
    """
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        return_dict=True,
    )
    return {k: v.to(model.device) for k, v in enc.items()}


@torch.inference_mode()
def generate(messages: list[dict], max_new_tokens: int = 120) -> str:
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
    """The pre-decision state: every layer's hidden vector at the last token
    of the prompt, from a clean no-cache forward pass (not from generate()).
    Returned as one tensor, shape [num_layers + 1, hidden_dim], on CPU in
    fp16 to keep the files small — this is a saved artifact, not a live
    tensor we're about to do more math with.
    """
    enc = build_ids(messages)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    # hidden_states: tuple of (num_layers + 1) tensors, each [1, seq_len, hidden_dim]
    # (+1 because index 0 is the embedding output, before any transformer layer)
    last_token = torch.stack([h[0, -1, :] for h in out.hidden_states])
    return last_token.to(torch.float16).cpu()


# ------------------------------------------------------------------ main ---
p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=620,
               help="max rows to process (default: all ~310 rows, twice over "
                    "to match 00_ascii.py's --n convention)")
args = p.parse_args()

df = pd.concat([load_config(CONFIG, "train"), load_config(CONFIG, "validation")])
sanity_check_labels(df, CONFIG)
df = df[df["task"].str.contains("ascii", case=False, na=False)]
print(f"\n{len(df)} ASCII rows available")

# did_lie is degenerate in this config (see HANDOFF.md / chat history) — every
# row is labeled False, so there's no real "balanced" split to sample here.
# Just take up to --n rows straight, same fixed seed as 00_ascii.py for the
# subset to be comparable if you run both.
sample = df.sample(min(args.n, len(df)), random_state=17)
print(f"Sampling {len(sample)} rows")

# --- part 0: capability control (text only, no activations needed) ---------
print("\n=== PART 0: capability control ===")
proof = generate([{"role": "user", "content": CAPABILITY_CHECK}], max_new_tokens=300)
print(proof)
(RESULTS / "capability_proof_local.txt").write_text(
    f"prompt:\n{CAPABILITY_CHECK}\n\nmodel: {TARGET_MODEL} (local, bf16)\n\n"
    f"response:\n{proof}\n")

done = load_done(OUT_PATH)
todo = sample[~sample["sample_id"].isin(done)]
print(f"{len(done)} already done, {len(todo)} to go")

# No ThreadPoolExecutor here on purpose: unlike the OpenRouter version, the
# bottleneck is one shared GPU, not network latency. Concurrent Python threads
# wouldn't run forward passes in parallel anyway (single CUDA context) — they'd
# just interleave and risk out-of-memory. One row at a time, straight through.
for _, row in tqdm(list(todo.iterrows()), total=len(todo)):
    uid = row["sample_id"]
    msgs = to_messages(row["trace"])
    probe = dict(row["lie_detection_prompt"])
    probe_text = probe["content"]

    try:
        # PART 1 — the rerun, plus its pre-decision state
        msgs_probe = msgs + [probe]
        a1 = generate(msgs_probe)
        lied = claimed_inability(a1, probe_text)
        lied_state = capture_state(msgs_probe)

        # PART 2 — the follow-up, plus its pre-decision state
        msgs_followup = msgs_probe + [
            {"role": "assistant", "content": a1},
            {"role": "user", "content": FOLLOWUP},
        ]
        a2 = generate(msgs_followup)
        admitted = admitted_lying(a2)
        doubled_down_state = capture_state(msgs_followup)
    except Exception as e:
        append(OUT_PATH, {"uid": uid, "error": str(e)})
        continue

    torch.save(
        {"sample_id": uid, "lied_state": lied_state,
         "doubled_down_state": doubled_down_state},
        ACTIVATIONS_DIR / f"{uid}.pt",
    )
    append(OUT_PATH, {
        "uid": uid,
        "dataset_did_lie": bool(row["did_lie"]),
        "lied": lied,
        "admitted": admitted,
        "probe_answer": a1,
        "followup_answer": a2,
        "task": row.get("task"),
        "activations_file": f"activations/{uid}.pt",
    })

# --------------------------------------------------------------- results ---
res = pd.DataFrame([r for r in read_jsonl(OUT_PATH) if "error" not in r])


def wilson(k, n):
    if n == 0:
        return "n/a"
    lo, hi = proportion_confint(k, n, method="wilson")
    return f"{k/n:6.1%}  [{lo:.1%}, {hi:.1%}]  n={n}"


ok1 = res.dropna(subset=["lied"])
print(f"\n=== PART 1b: gemma's lying rate, THIS RUN ===")
print(f"  {wilson(int(ok1['lied'].sum()), len(ok1))}")
print(f"  unparseable: {int(res['lied'].isna().sum())}")

both = ok1.dropna(subset=["admitted"])
lied = both[both["lied"]]
told = both[~both["lied"]]

print("\n=== PART 2: did it own up? ===")
print(f"  DOUBLING DOWN    denies | it lied      = "
      f"{wilson(int((~lied['admitted']).sum()), len(lied))}")
print(f"  confession       admits | it lied      = "
      f"{wilson(int(lied['admitted'].sum()), len(lied))}")
print(f"  false confession admits | truthful     = "
      f"{wilson(int(told['admitted'].sum()), len(told))}")

res.to_csv(RESULTS / "ascii_results.csv", index=False)
print(f"\nText table: {RESULTS / 'ascii_results.csv'}")
print(f"Activation files: {ACTIVATIONS_DIR}/<sample_id>.pt "
      f"(dict with 'lied_state', 'doubled_down_state', each "
      f"[num_layers+1, hidden_dim])")
