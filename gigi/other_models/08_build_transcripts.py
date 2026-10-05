"""Transcripts for the third-person monitors: each conversation up to and
including the model's answer, with no follow-up question (as for gemma's
third_person/results/transcripts.jsonl).

  - the seven follow-up categories: the conversation src/rollouts.py rebuilds
    for the row's `neutral` follow-up, minus the follow-up itself
  - the MASK subtypes: the pressured conversation + the model's statement

Only the categories passed (the ones this box ran). Appends; skips rows
already written.

Outputs: results/<MODEL_TAG>/transcripts.jsonl  {category, sample_id, track, messages}
Run:     python 08_build_transcripts.py --categories alibi,capture_secret
"""
import argparse
import json

from src import common as C
from src import mask_belief as MB
from src import rollouts as RO
from src.answer_probs import split_name

OUT = C.RESULTS / "transcripts.jsonl"

p = argparse.ArgumentParser()
p.add_argument("--categories", required=True)
args = p.parse_args()

done = {(r["category"], r["sample_id"]) for r in C.read_jsonl(OUT)} if OUT.exists() else set()
n = 0
for cat in args.categories.split(","):
    if cat.startswith("mask_") and cat != "mask_factual":
        sub = cat[len("mask_"):]
        ans = {r["sample_id"]: r for r in C.read_jsonl(C.RESULTS / f"mask_belief_{sub}_answers.jsonl")
               if "error" not in r and r.get("statement")}
        rows = [(r["sample_id"], None, r["messages"] + [{"role": "assistant", "content": ans[r["sample_id"]]["statement"]}])
                for r in MB.rows(sub) if r["sample_id"] in ans]
    else:
        rows = []
        for name, base, _followup, _answer, _kind in RO.BUILDERS[cat]():
            if name.endswith("__neutral"):
                sid, track, _ = split_name(name, cat)
                rows.append((sid, track, base))
    for sid, track, messages in rows:
        if (cat, sid) in done:
            continue
        C.append(OUT, {"category": cat, "sample_id": sid, "track": track, "messages": messages})
        done.add((cat, sid))
        n += 1
    print(f"[{cat}] {sum(1 for c, _ in done if c == cat)} transcripts", flush=True)
print(f"wrote {n} new transcripts -> {OUT}")
