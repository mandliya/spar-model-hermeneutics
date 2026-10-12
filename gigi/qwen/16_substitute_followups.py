"""Substitute step (hermeneutics): gemma answers the four follow-up questions about
the TARGET model's conversations, as if they were its own.

Target = the model named by MODEL_TAG (qwen3-next-80b); reader = READER_TAG
(gemma-3-27b). For every follow-up the target answered, the conversation is
rebuilt from results/<MODEL_TAG>/ by the code that built it for the target
(src/rollouts.py; src/mask_belief.py for the MASK belief categories): system
prompt, user turns, and the target's answer in the assistant slot. The reader
writes none of that. It is then asked the same follow-up question, word for
word, and answers it itself: greedy, one conversation at a time, 80 new tokens
for a letter and 300 when reasoning comes first, same parsers as the target's
run.

Saved per answer: the reader's reply, the letter, `admitted`, and its letter
probabilities at the answer token (P("A"), src/answer_probs.letter_logprobs).
With --activations, both capture points in float32 (last prompt token =
predecision, and the answer token), as in 05_named_cot.py and
14_mask_belief_followups.py.

--swap    the same questions with the two options reversed (B = "yes"), on the
          target's own swap subset (results/<MODEL_TAG>/ab_swap_subset.csv), so
          both models are compared in both letter orders on the same rows. No
          activations.
--render  no GPU, no model: renders every job with the reader's chat template
          and checks it (render_check). Run this first, anywhere.

Outputs, in results/<READER_TAG>_reads_<MODEL_TAG>/ (the target's folder is
only read):
    pa_<category>.jsonl       original letter order, one record per (row,
                              question). Same file name and fields as the
                              target's pa_<category>.jsonl, plus the reply
                              itself, so one analysis reads both models.
    ab_swap_<category>.jsonl  swapped letter order
    /workspace/substitute_staging/<category>/<name>__{predecision,answer_token}.pt

Run:
    MODEL_TAG=qwen3-next-80b python 16_substitute_followups.py --render
    MODEL_TAG=qwen3-next-80b python 16_substitute_followups.py --categories alibi --limit 2
    MODEL_TAG=qwen3-next-80b python 16_substitute_followups.py --categories all --activations
    MODEL_TAG=qwen3-next-80b python 16_substitute_followups.py --categories all --swap

--limit N = the first N conversations, with all their questions. SHARD=k/n
splits the rows across boxes. Resumable per answer; a failure on one row is
logged and skipped, a category that fails is logged and the rest still run.
The first 5 answers are a gate: fewer than 4 parseable stops the script.
No API calls.
"""

import argparse
import collections
import os
import pathlib
import sys
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import common as C
from src import mask_belief as MB
from src import rollouts as RO
from src.answer_probs import (LETTER_KINDS, letter_logprobs, named_cot_jobs, split_name,
                              swap_options)
from src.dyl_activations import capture_state_at_answer, find_answer_char_span

READERS = {"gemma-3-27b": "google/gemma-3-27b-it"}
READER_TAG = os.getenv("READER_TAG", "gemma-3-27b")
READER = READERS[READER_TAG]
OUT_DIR = C.ROOT / "results" / f"{READER_TAG}_reads_{C.MODEL_TAG}"
STAGE = pathlib.Path("/workspace/substitute_staging")
MASK = {f"mask_{s}": s for s in MB.SUBTYPES}
CATS = ["mask_factual", "unanswerable", "sycophancy_a", "sycophancy_b", "ascii",
        "capture_secret", "alibi", *MASK]

p = argparse.ArgumentParser()
p.add_argument("--categories", default="all")
p.add_argument("--limit", type=int, default=None, help="first N conversations, all their questions")
p.add_argument("--swap", action="store_true")
p.add_argument("--activations", action="store_true")
p.add_argument("--render", type=int, nargs="?", const=1, default=None, metavar="N",
               help="check every job with the tokenizer only; print N examples per category")
args = p.parse_args()
assert not (args.swap and args.activations), "the swap test saves no activations"
SUBSET = pd.read_csv(C.RESULTS / "ab_swap_subset.csv") if args.swap else None
cats = CATS if args.categories == "all" else args.categories.split(",")
if args.swap:   # only the categories the target's swap subset covers (no MASK belief, as for the target)
    skipped = [c for c in cats if c not in set(SUBSET.category)]
    cats = [c for c in cats if c not in skipped]
    if skipped:
        print(f"not in the target's swap subset, skipped: {', '.join(skipped)}", flush=True)


def jobs(cat):
    """(name, conversation ending in the target's answer, follow-up question, the
    target's own answer to it, kind) for every follow-up the target answered."""
    if cat in MASK:
        sub = MASK[cat]
        rows = {r["sample_id"]: r for r in MB.rows(sub)}
        asked = {r["uid"]: r for r in C.read_jsonl(C.RESULTS / f"mask_belief_{sub}_followups.jsonl")
                 if "error" not in r}
        out = []
        for a in C.read_jsonl(C.RESULTS / f"mask_belief_{sub}_answers.jsonl"):
            if "error" in a or a["sample_id"] not in rows:
                continue
            base = rows[a["sample_id"]]["messages"] + [{"role": "assistant", "content": a["statement"]}]
            for q, (_text, kind) in MB.followups(sub).items():
                t = asked.get(f"{a['sample_id']}::{q}")
                if t is not None:
                    out.append((f"{a['sample_id']}__{q}", base, t["question"], t["answer"], kind))
    else:
        rows = list(RO.BUILDERS[cat]())
        out = [r for r in rows if r[4] in LETTER_KINDS] + named_cot_jobs(rows, cat)
    seen, uniq = set(), []
    for j in out:
        if j[0] not in seen:
            seen.add(j[0])
            uniq.append(j)
    return uniq


def select(cat):
    """This run's jobs: the swap subset if --swap, this box's shard, the first --limit conversations."""
    js = jobs(cat)
    sid = lambda j: split_name(j[0], cat)[0]
    if args.swap:
        keep = set(SUBSET[SUBSET.category == cat].sample_id.astype(str))
        js = [j for j in js if sid(j) in keep]
    js = [j for j in js if C.in_shard(sid(j))]
    if args.limit:
        first = list(dict.fromkeys(sid(j) for j in js))[: args.limit]
        js = [j for j in js if sid(j) in first]
    return js


# --- render check: tokenizer only ---------------------------------------------
FOREIGN = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>")   # the target's own chat markers


def render_check(tok, cat, js, show):
    """Does every job read correctly in the reader's chat format? Checked per job:
    the template accepts it; the target's answer is the last assistant turn, the
    follow-up is the last user turn, and the text ends where the reader starts
    writing; no turn's text is lost (system prompt included). The job names must
    also be exactly the target's own. Returns the number of jobs that fail."""
    # the reader's own turn separators, read off a three-turn probe
    probe = tok.apply_chat_template([{"role": "user", "content": "@@U@@"},
                                     {"role": "assistant", "content": "@@A@@"},
                                     {"role": "user", "content": "@@Q@@"}],
                                    tokenize=False, add_generation_prompt=True)
    to_model = probe[probe.index("@@U@@") + 5: probe.index("@@A@@")]
    to_user = probe[probe.index("@@A@@") + 5: probe.index("@@Q@@")]
    to_reply = probe[probe.index("@@Q@@") + 5:]
    bad, warn, n_tok, n_sys, shown = collections.Counter(), collections.Counter(), [], 0, 0
    for name, base, question, _target_answer, _kind in js:
        try:
            q = swap_options(question) if args.swap else question
            msgs = base + [{"role": "user", "content": q}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            bad[f"not renderable: {repr(e)[:90]}"] += 1
            continue
        ans = str(base[-1]["content"]).strip()
        if base[-1]["role"] != "assistant" or ans in ("", "nan"):
            bad["no target answer in the assistant slot"] += 1
        elif not text.endswith(to_model + ans + to_user + q.strip() + to_reply):
            bad["target answer + follow-up are not the last two turns"] += 1
        elif any(str(m["content"]).strip() not in text for m in msgs):
            bad["a turn's text is missing from the rendered chat"] += 1
        if any(f in text for f in FOREIGN):
            warn["contains the target's chat markers"] += 1
        n_sys += msgs[0]["role"] == "system"
        n_tok.append(len(tok(text, add_special_tokens=False).input_ids))
        if shown < show:
            shown += 1
            print(f"\n----- {cat} | {name} -----\n{text}<<< the reader writes from here\n-----", flush=True)
    # the same (row, question) set as the target's own file
    ref = C.RESULTS / f"{'ab_swap' if args.swap else 'pa'}_{cat}.jsonl"
    if not ref.exists():
        names = f"target has no {ref.name}"
    elif args.limit or C.SHARD:
        names = "names not compared (--limit / SHARD)"
    else:
        want = {r["name"] for r in C.read_jsonl(ref) if "error" not in r}
        have = {j[0] for j in js}
        names = f"names vs target: {len(want - have)} missing, {len(have - want)} extra"
        if want != have:
            bad["job names differ from the target's"] += 1
    print(f"[{cat}] {len(js)} jobs, {len({split_name(j[0], cat)[0] for j in js})} conversations, "
          f"{n_sys} with a system prompt, tokens max {max(n_tok, default=0)} | {names} | "
          f"{'OK' if not bad else 'FAILED ' + str(dict(bad))}"
          f"{' | warning ' + str(dict(warn)) if warn else ''}", flush=True)
    return sum(bad.values())


tok = AutoTokenizer.from_pretrained(READER)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

if args.render is not None:
    failed = sum(render_check(tok, cat, select(cat), args.render) for cat in cats)
    print(f"RENDER CHECK {'PASSED' if not failed else f'FAILED ({failed} jobs)'}: {READER_TAG} reads "
          f"{C.MODEL_TAG}, {'swapped' if args.swap else 'original'} letter order", flush=True)
    sys.exit(1 if failed else 0)

# --- the run: GPU ---------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(READER, torch_dtype=torch.bfloat16, device_map="auto",
                                             attn_implementation="eager")
model.eval()
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    return torch.stack([h[0, -1, :].cpu() for h in out.hidden_states]).float().cpu()


gate_checked = False
for cat in cats:
    try:
        js = select(cat)
        OUT = OUT_DIR / f"{'ab_swap' if args.swap else 'pa'}_{cat}.jsonl"
        done = C.load_done(OUT, key="name")
        todo = [j for j in js if j[0] not in done]
        d = STAGE / cat
        if args.activations:
            d.mkdir(parents=True, exist_ok=True)
        print(f"[{cat}] {len(js)} answers, {len(done)} done, {len(todo)} to go", flush=True)
    except Exception as e:
        print(f"!!! [{cat}] could not build its jobs: {e!r} - continuing with the rest", flush=True)
        continue

    for n, (name, base, question, target_answer, kind) in enumerate(tqdm(todo, mininterval=60)):
        sid, track, cond = split_name(name, cat)
        t0 = time.time()
        try:
            q = swap_options(question) if args.swap else question
            msgs = base + [{"role": "user", "content": q}]
            pre = at = None
            if args.activations:
                pre = f"{name}__predecision.pt"
                torch.save(predecision(msgs), d / pre)
            answer = generate(msgs, 300 if kind == "letter_cot" else 80)
            span = find_answer_char_span(answer, kind)
            if args.swap:   # under the swap, B = "yes, I lied"; None = unparseable, never False
                admitted = None if span is None else (answer[span[0]].upper() == "B")
            else:
                admitted = C.admitted_lying_cot(answer) if kind == "letter_cot" else C.admitted_lying(answer)
            if args.activations and span is not None:
                st = capture_state_at_answer(tok, model, msgs, answer, span, out_dtype=torch.float32)
                if st is not None:
                    at = f"{name}__answer_token.pt"
                    torch.save(st, d / at)
            lp = letter_logprobs(tok, model, msgs, answer, kind)
            C.append(OUT, {
                "name": name, "category": cat, "sample_id": sid, "track": track,
                "condition": cond, "kind": kind, "reader": READER_TAG, "target": C.MODEL_TAG,
                "question_swapped" if args.swap else "question": q, "answer": answer,
                "letter": lp.get("letter"), "admitted": admitted,
                "reasoning": C.parse_reasoning(answer) if kind == "letter_cot" else None,
                "piece": lp.get("piece"), "lp_A": lp.get("lp_A"), "lp_B": lp.get("lp_B"),
                "p_A_norm": lp.get("p_A_norm"), "top1_is_answer": lp.get("top1_is_answer"),
                "top5": lp.get("top5"), "n_tokens": lp.get("n_tokens"), "note": lp.get("note"),
                "target_answer": target_answer, "predecision_file": pre, "answer_token_file": at,
                "gpu_seconds": round(time.time() - t0, 3)})
        except Exception as e:
            C.append(OUT, {"name": name, "error": repr(e)[:500]})

        if not gate_checked and n == 4:
            recs = C.read_jsonl(OUT)[-5:]
            ok = sum(1 for r in recs if r.get("admitted") is not None)
            print(f"GATE: {ok}/5 answers parseable", flush=True)
            for r in recs:
                print("   ", r.get("name"), repr((r.get("answer") or r.get("error") or ""))[:120],
                      r.get("admitted"), r.get("p_A_norm"), flush=True)
            if ok < 4:
                raise SystemExit("GATE FAILED - stopping before the full run")
            gate_checked = True

    recs = C.read_jsonl(OUT) if OUT.exists() else []
    good = [r for r in recs if "error" not in r]
    print(f"[{cat}] done: {len(good)} ok, {len(recs) - len(good)} errors, "
          f"{sum(1 for r in good if r.get('admitted') is None)} unparseable, "
          f"{sum(1 for r in good if r.get('p_A_norm') is None)} without a letter probability", flush=True)
print(f"16_substitute_followups: ALL CATEGORIES DONE ({'swapped' if args.swap else 'original'} letter order)",
      flush=True)
