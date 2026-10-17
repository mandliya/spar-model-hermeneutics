"""Make every first-run activation float32 (the originals were float16).

Part 1 - recapture, rebuilding each first-run prompt exactly as the original
script did, from the saved results (no generation, no API calls):
  ascii           <uid>.pt = {"sample_id", "lied_state", "doubled_down_state"}
                    lied_state: trace + capability probe
                    doubled_down_state: + gemma's saved probe answer + FOLLOWUP
  sycophancy A    <uid>__neutral.pt   trace + the bare question
                  <uid>__pressured.pt trace + the probe (user's belief stated)
  sycophancy B    <uid>.pt            trace + the probe
  unanswerable    <file>__claim__predecision.pt, __claim__answer_token.pt
                    (answer token teacher-forced from the saved claim reply)
  capture_secret  <file>__capture__predecision.pt   trace + the probe
  alibi           <uid>__alibi__predecision.pt      the first 500 rows only
                    (the other 781 were captured in float32 already)
A sample per category is compared with its float16 original (cosine on finite
values); a category that doesn't match is not uploaded.

Part 2 - only if every category matched:
  archive/first_run_fp16/<category>/  <- every float16 first-run file (server-side
                                         copy, sha256-verified, then removed from first_run/)
  first_run/<category>/               <- the float32 files
  mask_factual: first_run/mask_factual/fp32/* moved up to first_run/mask_factual/
"""
import os
import pathlib
import random
import sys

import pandas as pd
import torch
from huggingface_hub import CommitOperationCopy, CommitOperationDelete, HfApi, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import common as C
from src import rollouts as RO
from src.dyl_activations import capture_state_at_answer, find_answer_char_span
from src.hf_sync import upload

R = "model-hermeneutics/gigi-lie-detection-activations"
api = HfApi(token=os.environ["HF_TOKEN"])
RES = C.RESULTS
OUT = pathlib.Path("/workspace/first_run_fp32")
say = lambda *a: print(*a, flush=True)

tok = AutoTokenizer.from_pretrained(C.TARGET_MODEL)
model = AutoModelForCausalLM.from_pretrained(C.TARGET_MODEL, torch_dtype=torch.bfloat16,
                                             device_map="auto", attn_implementation="eager")
model.eval()


@torch.inference_mode()
def pre(messages):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False)
    return torch.stack([h[0, -1, :] for h in out.hidden_states]).float().cpu()


def msgs_of(ds, sid):
    r = ds.loc[sid]
    return C.to_messages(r["trace"]), dict(r["lie_detection_prompt"])


def save(cat, name, obj):
    d = OUT / cat
    d.mkdir(parents=True, exist_ok=True)
    torch.save(obj, d / name)


# ---------------------------------------------------------------- Part 1 ---
jobs = {}   # category -> list of (filename, callable producing the object)

ds = RO.dataset("google-gemma-3-27b-it_task-group_sandbagging",
                lambda d: d["task"].str.contains("ascii", case=False, na=False))
a = pd.read_csv(RES / "ascii_results.csv")
def ascii_job(uid, a1):
    def f():
        m, p = msgs_of(ds, uid)
        return {"sample_id": uid, "lied_state": pre(m + [p]),
                "doubled_down_state": pre(m + [p, {"role": "assistant", "content": str(a1)},
                                                {"role": "user", "content": C.FOLLOWUP}])}
    return f
jobs["ascii"] = [(f"{r.uid}.pt", ascii_job(r.uid, r.probe_answer)) for r in a.itertuples()
                 if r.uid in ds.index]

dsy = RO.dataset("google-gemma-3-27b-it_task-group_sycophancy", lambda d: d["task"].notna())
sa = pd.read_csv(RES / "sycophancy_answer_conditions.csv")
sb = pd.read_csv(RES / "sycophancy_feedback_conditions.csv")
jobs["sycophancy"] = []
for r in sa.itertuples():
    if r.uid in dsy.index:
        jobs["sycophancy"].append((f"{r.uid}__neutral.pt",
            (lambda u=r.uid, q=r.question: pre(msgs_of(dsy, u)[0] + [{"role": "user", "content": str(q)}]))))
        jobs["sycophancy"].append((f"{r.uid}__pressured.pt",
            (lambda u=r.uid: (lambda m, p: pre(m + [p]))(*msgs_of(dsy, u)))))
for r in sb.itertuples():
    if r.uid in dsy.index:
        jobs["sycophancy"].append((f"{r.uid}.pt", (lambda u=r.uid: (lambda m, p: pre(m + [p]))(*msgs_of(dsy, u)))))

dun = RO.dataset("google-gemma-3-27b-it_task-group_unanswerable", lambda d: d["task"].notna())
un = pd.read_csv(RES / "unanswerable_claims.csv")
jobs["unanswerable"] = []
for r in un.itertuples():
    if r.sample_id not in dun.index:
        continue
    jobs["unanswerable"].append((os.path.basename(r.predecision_activations_file),
        (lambda s=r.sample_id: (lambda m, p: pre(m + [p]))(*msgs_of(dun, s)))))
    if isinstance(r.answer_token_activations_file, str):
        def at(s=r.sample_id, reply=str(r.claim_reply)):
            m, p = msgs_of(dun, s)
            span = find_answer_char_span(reply, "answer_tag")
            return None if span is None else capture_state_at_answer(
                tok, model, m + [p], reply, span, out_dtype=torch.float32)
        jobs["unanswerable"].append((os.path.basename(r.answer_token_activations_file), at))

dg = RO.dataset("google-gemma-3-27b-it_task-group_games", lambda d: d["task"].notna())
cs = pd.read_csv(RES / "capture_secret_conditions.csv", low_memory=False).drop_duplicates("sample_id")
jobs["capture_secret"] = [(os.path.basename(r.predecision_activations_file),
                           (lambda s=r.sample_id: (lambda m, p: pre(m + [p]))(*msgs_of(dg, s))))
                          for r in cs.itertuples() if r.sample_id in dg.index]
first500 = set(pd.read_csv(RES / "alibi_subset.csv").sample_id)
al = pd.read_csv(RES / "alibi_conditions.csv").drop_duplicates("sample_id")
jobs["alibi"] = [(f"{r.sample_id}__alibi__predecision.pt",
                  (lambda s=r.sample_id: (lambda m, p: pre(m + [p]))(*msgs_of(dg, s))))
                 for r in al.itertuples() if r.sample_id in first500 and r.sample_id in dg.index]

for cat, js in jobs.items():
    done = 0
    for name, f in js:
        if (OUT / cat / name).exists():
            continue
        obj = f()
        if obj is not None:
            save(cat, name, obj)
            done += 1
    say(f"[{cat}] {len(js)} tensors expected, {len(list((OUT / cat).glob('*.pt')))} saved")

# compare a sample with the float16 originals on HF
ORIG = {"ascii": lambda n: f"first_run/ascii/{n}",
        "unanswerable": lambda n: f"first_run/unanswerable/{n}",
        "capture_secret": lambda n: f"first_run/capture_secret/{n}",
        "alibi": lambda n: f"first_run/alibi/{n}",
        "sycophancy": lambda n: f"first_run/sycophancy/{n[:-3]}__main.pt" if "__" not in n
                                 else f"first_run/sycophancy/{n}"}
have = {s.rfilename for s in api.repo_info(R, repo_type="dataset").siblings}
ok_all = True


def flat(x):
    return torch.cat([x["lied_state"].flatten(), x["doubled_down_state"].flatten()]) if isinstance(x, dict) else x.flatten()


for cat in jobs:
    files = sorted((OUT / cat).glob("*.pt"))
    random.Random(0).shuffle(files)
    worst, n = 1.0, 0
    for f in files:
        path = ORIG[cat](f.name)
        if path not in have:
            continue
        old = flat(torch.load(hf_hub_download(R, path, repo_type="dataset"), weights_only=False)).float()
        new = flat(torch.load(f, weights_only=False)).float()
        fin = torch.isfinite(old)
        cos = torch.nn.functional.cosine_similarity(new[fin], old[fin], dim=0).item()
        worst, n = min(worst, cos), n + 1
        if n == 5:
            break
    good = n > 0 and worst >= 0.999
    ok_all &= good
    say(f"CHECK {cat}: {n} compared with the float16 originals, worst cosine {worst:.6f} -> {'OK' if good else 'FAIL'}")

if not ok_all:
    say("!!! a category did not match its originals - Part 2 skipped, nothing on HF changed")
    sys.exit(1)

# ---------------------------------------------------------------- Part 2 ---
info = {s.rfilename: s for s in api.repo_info(R, repo_type="dataset", files_metadata=True).siblings}
start = api.repo_info(R, repo_type="dataset").sha
rest781 = set(pd.read_csv(RES / "alibi_rest.csv").sample_id)
moves = {}
for f in info:
    if not f.startswith("first_run/") or "/fp32/" in f:
        continue
    if f.startswith("first_run/alibi/") and os.path.basename(f).split("__")[0] in rest781:
        continue                                    # already float32
    moves[f] = "archive/first_run_fp16/" + f[len("first_run/"):]
say(f"archiving {len(moves)} float16 first-run files (starting revision {start[:8]})")
api.create_commit(R, repo_type="dataset", commit_message="Archive the float16 first-run activations (copy)",
                  operations=[CommitOperationCopy(src_path_in_repo=s, path_in_repo=d)
                              for s, d in moves.items() if d not in info])
now = {s.rfilename: s for s in api.repo_info(R, repo_type="dataset", files_metadata=True).siblings}
bad = [d for s, d in moves.items() if d not in now or now[d].lfs.sha256 != info[s].lfs.sha256]
say(f"archive copies verified: {len(moves) - len(bad)}/{len(moves)}")
if bad:
    say("!!! archive copy check failed - float16 originals left in place, float32 not uploaded")
    sys.exit(1)
api.create_commit(R, repo_type="dataset",
                  commit_message=f"first_run: remove float16 files now in archive/first_run_fp16 (recoverable at {start[:8]})",
                  operations=[CommitOperationDelete(path_in_repo=s) for s in moves])

for cat in jobs:
    upload(OUT / cat, f"first_run/{cat}", folder=True)

mask32 = [f for f in now if f.startswith("first_run/mask_factual/fp32/")]
api.create_commit(R, repo_type="dataset", commit_message="first_run: mask_factual float32 files up one level",
                  operations=[CommitOperationCopy(src_path_in_repo=f, path_in_repo="first_run/mask_factual/" + os.path.basename(f))
                              for f in mask32])
api.create_commit(R, repo_type="dataset", commit_message="first_run: remove mask_factual/fp32/ (now one level up)",
                  operations=[CommitOperationDelete(path_in_repo=f) for f in mask32])

final = [s.rfilename for s in api.repo_info(R, repo_type="dataset").siblings if s.rfilename.startswith("first_run/")]
from collections import Counter
say("first_run/ now:", dict(Counter(f.split("/")[1] for f in final)))
for cat in jobs:
    local = {f"first_run/{cat}/{p.name}" for p in (OUT / cat).glob("*.pt")}
    say(f"VERIFY first_run/{cat}: {len(local & set(final))}/{len(local)} float32 files on HF")
say("FIRST RUN FP32 DONE")
