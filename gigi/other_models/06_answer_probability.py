"""P("A"): how sure gemma was when it answered each A/B follow-up question.

For every saved answer to the 11 A/B follow-up wordings (the 7 direct ones,
the 3 original *_cot ones and named_cot), in every category: rebuild the exact
conversation (src/rollouts.py, the same builder the float32 recapture used),
teacher-force gemma's saved answer, and read its next-token distribution at the
position that produced the answer letter. Nothing is generated.

Per answer it saves the letter gemma wrote, the log-probabilities of "A" and
"B" at that position (the same token form, e.g. "▁A" vs "▁B"), the top 5
tokens, and p_A_norm = P(A) / (P(A) + P(B)). In every original wording A means
"yes, I lied", so p_A_norm is the score an AUROC is computed from.

Built-in check: at temperature 0 gemma wrote its most likely token, so the
answer token should be the top-1 token here as well (`top1_is_answer`). The
first 5 rows of the first category are a gate: fewer than 4 passing stops the
script before the full run.

The two confidence wordings are not here: their 0-100 score is already a
continuous score.

Outputs:  results/pa_<category>.jsonl, uploaded to HF after each category
          (results/answer_probability/).

Run:
    python 06_answer_probability.py --categories mask_factual --limit 5
    python 06_answer_probability.py --categories all

Resumable per answer; a failure on one row is logged and skipped.
"""

import argparse
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import load_tokenizer as _load_tokenizer
from src import common as C
from src import rollouts as RO
from src.answer_probs import LETTER_KINDS, letter_logprobs, named_cot_jobs, split_name
from src.hf_sync import upload

ORDER = ["mask_factual", "unanswerable", "sycophancy_a", "sycophancy_b", "ascii",
         "capture_secret", "alibi"]
p = argparse.ArgumentParser()
p.add_argument("--categories", default="all")
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()
cats = ORDER if args.categories == "all" else args.categories.split(",")

tok = _load_tokenizer()
model = AutoModelForCausalLM.from_pretrained(
    C.TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager")
model.eval()

gate_checked = False
for cat in cats:
    rows = list(RO.BUILDERS[cat]())
    jobs = [r for r in rows if r[4] in LETTER_KINDS] + named_cot_jobs(rows, cat)
    seen, uniq = set(), []
    for j in jobs:
        if j[0] not in seen:
            seen.add(j[0])
            uniq.append(j)
    jobs = uniq[: args.limit] if args.limit else uniq
    OUT = C.RESULTS / f"pa_{cat}.jsonl"
    done = C.load_done(OUT, key="name")
    todo = [j for j in jobs if j[0] not in done]
    print(f"[{cat}] {len(jobs)} answers, {len(done)} done, {len(todo)} to go", flush=True)

    for n, (name, base, question, answer, kind) in enumerate(tqdm(todo, mininterval=60)):
        sid, track, cond = split_name(name, cat)
        t0 = time.time()
        try:
            rec = letter_logprobs(tok, model, base + [{"role": "user", "content": question}],
                                  answer, kind)
            C.append(OUT, {"name": name, "category": cat, "sample_id": sid, "track": track,
                           "condition": cond, "kind": kind, "answer": answer, **rec,
                           "gpu_seconds": round(time.time() - t0, 3)})
        except Exception as e:
            C.append(OUT, {"name": name, "error": repr(e)[:500]})

        if not gate_checked and n == 4:
            recs = [r for r in C.read_jsonl(OUT)][-5:]
            ok = sum(1 for r in recs if r.get("top1_is_answer") and r.get("p_A_norm") is not None)
            print(f"GATE: {ok}/5 rows with a readable letter that is gemma's top-1 token", flush=True)
            for r in recs:
                print("   ", r.get("name"), r.get("letter"), r.get("piece"), r.get("p_A_norm"),
                      r.get("top1_is_answer"), r.get("note") or r.get("error") or "", flush=True)
            if ok < 4:
                raise SystemExit("GATE FAILED - stopping before the full run")
            gate_checked = True

    recs = C.read_jsonl(OUT)
    good = [r for r in recs if "error" not in r]
    print(f"[{cat}] done: {len(good)} ok, {len(recs) - len(good)} errors, "
          f"{sum(1 for r in good if r.get('p_A_norm') is None)} without a letter probability, "
          f"{sum(1 for r in good if r.get('top1_is_answer') is False)} where the letter was not top-1",
          flush=True)
    if not args.limit:
        upload(OUT, f"results/answer_probability/pa_{cat}.jsonl")
print("06_answer_probability: ALL CATEGORIES DONE", flush=True)
