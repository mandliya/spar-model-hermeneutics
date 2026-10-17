"""Shared by 06_answer_probability.py and 07_ab_swap.py.

letter_logprobs(): teacher-force a saved answer and read gemma's next-token
distribution at the exact position that produced the answer letter. Nothing is
generated. Uses the same char-span -> token mapping as
dyl_activations.capture_state_at_answer, so the position is the same token the
answer_token activations were taken at.

swap_options(): reverse the two options of any A/B follow-up wording, leaving
every other character unchanged.

named_cot_jobs(): the named_cot rows, rebuilt from the `named` conversations
and the exact question text saved with each named_cot answer.
"""

import json
import math
import re

import torch

from src import common as C
from src.dyl_activations import find_answer_char_span

LETTER_KINDS = ("letter", "letter_cot")

# "A: Yes, I lied\nB: No, I was truthful" / "A: Yes\nB: No" (letter wordings)
# "A: Yes, I lied   B: No, I was truthful" / "A: Yes   B: No" (cot wordings)
_OPTIONS_RE = re.compile(r"A: (Yes[^\n]*?)(\n|   )B: (No[^\n]*?)(?=\n|$)")


def swap_options(text: str) -> str:
    ms = list(_OPTIONS_RE.finditer(text))
    if len(ms) != 1:
        raise ValueError(f"expected one A/B option pair, found {len(ms)}: {text[:120]!r}")
    m = ms[0]
    yes, sep, no = m.group(1), m.group(2), m.group(3)
    out = text[:m.start()] + f"A: {no}{sep}B: {yes}" + text[m.end():]
    assert out != text and len(out) == len(text)
    return out


def split_name(name: str, category: str):
    """(sample_id, track, condition) from a job name."""
    stem, cond = name.rsplit("__", 1)
    track = None
    if category.startswith("sycophancy"):
        track, stem = stem.split("__", 1)
    return stem, track, cond


def named_cot_jobs(builder_rows, category: str):
    """(name, base, question, answer, kind) for every saved named_cot answer."""
    base_by = {name[: -len("__named")]: base for name, base, _f, _a, _k in builder_rows
               if name.endswith("__named")}
    out = []
    p = C.RESULTS / f"named_cot_{category}.jsonl"
    for line in p.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r:
            continue
        stem = r["name"][: -len("__named_cot")]
        if stem in base_by:
            out.append((r["name"], base_by[stem], r["question"], r["answer"], "letter_cot"))
    return out


@torch.inference_mode()
def letter_logprobs(tok, model, messages, answer: str, kind: str):
    """Log-probs of the answer letter and of the other letter, at the position
    that produced the letter. Returns a dict; letter fields are None (never
    guessed) if no letter can be found or mapped to a token."""
    span = find_answer_char_span(answer, kind)
    if span is None:
        return {"letter": None, "note": "no answer letter found"}
    letter = answer[span[0]].upper()
    other = "B" if letter == "A" else "A"

    full = tok.apply_chat_template(messages + [{"role": "assistant", "content": answer}],
                                   tokenize=False, add_generation_prompt=False)
    rs = full.rfind(answer)
    if rs == -1:
        return {"letter": letter, "note": "answer text not found in rendered chat"}
    target = rs + span[0]
    enc = tok(full, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"][0].tolist()
    idx = next((i for i, (s, e) in enumerate(offsets) if s <= target < e), None)
    if idx is None or idx == 0:
        return {"letter": letter, "note": "letter not mapped to a token"}

    ids = enc["input_ids"][0]
    logits = model(input_ids=ids[:idx].unsqueeze(0).to(model.device),
                   use_cache=False).logits[0, -1].float()
    lp = torch.log_softmax(logits, dim=-1)

    ans_id = int(ids[idx])
    piece = tok.convert_ids_to_tokens(ans_id)
    other_id = None
    if piece.count(letter) == 1:
        other_piece = piece.replace(letter, other)
        oid = tok.convert_tokens_to_ids(other_piece)
        if oid is not None and oid != tok.unk_token_id and tok.convert_ids_to_tokens(oid) == other_piece:
            other_id = oid
    top = torch.topk(lp, 5)
    lp_letter = float(lp[ans_id])
    lp_other = None if other_id is None else float(lp[other_id])
    rec = {
        "letter": letter,
        "piece": piece,
        "lp_A": lp_letter if letter == "A" else lp_other,
        "lp_B": lp_letter if letter == "B" else lp_other,
        "top1_is_answer": int(top.indices[0]) == ans_id,
        "top5": [[tok.convert_ids_to_tokens(int(i)), round(float(v), 4)]
                 for v, i in zip(top.values, top.indices)],
        "n_tokens": idx,
    }
    if rec["lp_A"] is not None and rec["lp_B"] is not None:
        a, b = math.exp(rec["lp_A"]), math.exp(rec["lp_B"])
        rec["p_A_norm"] = a / (a + b)
    else:
        rec["p_A_norm"] = None
    return rec
