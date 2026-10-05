"""MASK belief categories, stage 3 (GPU): the four follow-up questions.

Asked right after gemma's pressured answer, on every row (labels are applied
at analysis time, so no row has to wait for the lie decision): neutral,
neutral_cot, named, named_cot (src/mask_belief.followups). Same token limits
and parsers as every other category; both capture points saved in float32;
gemma's letter probabilities read at the answer token (P("A"), A = yes).

Outputs: results/mask_belief_<subtype>_followups.jsonl
         /workspace/mask_belief_staging/followups/mask_<subtype>/<sid>__<question>__{predecision,answer_token}.pt
Run:  python 14_mask_belief_followups.py --subtypes all [--limit N]
The first 5 answers are a gate: fewer than 4 parseable stops the script.
"""
import argparse
import pathlib
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import load_tokenizer as _load_tokenizer
from src import common as C
from src import mask_belief as MB
from src.answer_probs import letter_logprobs
from src.dyl_activations import capture_state_at_answer, find_answer_char_span

p = argparse.ArgumentParser()
p.add_argument("--subtypes", default="all")
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()
subs = list(MB.SUBTYPES) if args.subtypes == "all" else args.subtypes.split(",")
STAGE = pathlib.Path("/workspace/mask_belief_staging/followups")

tok = _load_tokenizer()
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(C.TARGET_MODEL, torch_dtype=torch.bfloat16,
                                             device_map="auto", attn_implementation="eager")
model.eval()


def enc(messages):
    e = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                return_dict=True)
    return {k: v.to(model.device) for k, v in e.items()}


@torch.inference_mode()
def generate(messages, n):
    e = enc(messages)
    out = model.generate(**e, max_new_tokens=n, do_sample=False, temperature=None, top_p=None,
                         top_k=None, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][e["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.inference_mode()
def predecision(messages):
    out = model(**enc(messages), output_hidden_states=True, use_cache=False)
    return torch.stack([h[0, -1, :] for h in out.hidden_states]).float().cpu()


gate_checked = False
for sub in subs:
    rows = {r["sample_id"]: r for r in MB.rows(sub)}
    ans = [r for r in C.read_jsonl(C.RESULTS / f"mask_belief_{sub}_answers.jsonl") if "error" not in r]
    jobs = [(a, q) for a in ans for q in MB.followups(sub)]
    if args.limit:
        jobs = jobs[: args.limit]
    OUT = C.RESULTS / f"mask_belief_{sub}_followups.jsonl"
    done = C.load_done(OUT, key="uid")
    todo = [(a, q) for a, q in jobs if f"{a['sample_id']}::{q}" not in done]
    d = STAGE / f"mask_{sub}"
    d.mkdir(parents=True, exist_ok=True)
    print(f"[{sub}] {len(ans)} rows x 4 questions, {len(done)} done, {len(todo)} to go", flush=True)
    for n, (a, q) in enumerate(tqdm(todo, mininterval=60)):
        sid, (text, kind) = a["sample_id"], MB.followups(sub)[q]
        t0 = time.time()
        try:
            msgs = rows[sid]["messages"] + [{"role": "assistant", "content": a["statement"]},
                                            {"role": "user", "content": text}]
            torch.save(predecision(msgs), d / f"{sid}__{q}__predecision.pt")
            reply = generate(msgs, 300 if kind == "letter_cot" else 80)
            admitted = C.admitted_lying_cot(reply) if kind == "letter_cot" else C.admitted_lying(reply)
            at = None
            span = find_answer_char_span(reply, kind)
            if span is not None:
                st = capture_state_at_answer(tok, model, msgs, reply, span, out_dtype=torch.float32)
                if st is not None:
                    torch.save(st, d / f"{sid}__{q}__answer_token.pt")
                    at = f"{sid}__{q}__answer_token.pt"
            lp = letter_logprobs(tok, model, msgs, reply, kind) if admitted is not None else {}
            C.append(OUT, {"uid": f"{sid}::{q}", "sample_id": sid, "subtype": sub, "condition": q,
                           "question": text, "answer": reply, "admitted": admitted,
                           "reasoning": C.parse_reasoning(reply) if kind == "letter_cot" else None,
                           "lp_A": lp.get("lp_A"), "lp_B": lp.get("lp_B"),
                           "p_A_norm": lp.get("p_A_norm"), "top1_is_answer": lp.get("top1_is_answer"),
                           "predecision_file": f"{sid}__{q}__predecision.pt",
                           "answer_token_file": at, "gpu_seconds": round(time.time() - t0, 2)})
        except Exception as e:
            C.append(OUT, {"uid": f"{sid}::{q}", "sample_id": sid, "error": repr(e)[:500]})
        if not gate_checked and n == 4:
            last = C.read_jsonl(OUT)[-5:]
            ok = sum(1 for r in last if r.get("admitted") is not None)
            print(f"GATE: {ok}/5 follow-up answers parseable", flush=True)
            if ok < 4:
                raise SystemExit("GATE FAILED - stopping before the full run")
            gate_checked = True
    recs = C.read_jsonl(OUT)
    good = [r for r in recs if "error" not in r]
    print(f"[{sub}] follow-ups: {len(good)} ok, {len(recs) - len(good)} errors, "
          f"{sum(1 for r in good if r.get('admitted') is None)} unparseable", flush=True)
