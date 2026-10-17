"""Recapture mask-factual's first-run activations (all 86 rows).

Only 9 of the 86 original `<uid>.pt` files ever reached HF. This rebuilds the
exact state 00_mask_factual_provided.py saved: every layer's hidden state at
the last prompt token of trace + lie_detection_prompt, before gemma answers.
That state depends only on the prompt, so it can be recomputed exactly.

Saved twice:
  float16  <uid>.pt  - same format as the original files; the 77 missing ones
                        go to first_run/mask_factual/ next to the 9 existing
  float32  <uid>.pt  - all 86, to first_run/mask_factual/fp32/

Check: one of the 9 existing HF files is downloaded and compared with its
recapture before anything is uploaded. A large difference stops the script.

Run:
    python 08_mask_first_run_recapture.py
"""

import pathlib

import pandas as pd
import torch
from huggingface_hub import hf_hub_download, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import common as C
from src import rollouts as RO
from src.hf_sync import REPO, upload

STAGE = pathlib.Path("/workspace/fp32_staging/mask_first_run")
F16, F32 = STAGE / "fp16", STAGE / "fp32"
F16.mkdir(parents=True, exist_ok=True)
F32.mkdir(parents=True, exist_ok=True)

ds = RO.dataset("google-gemma-3-27b-it_task-group_mask-factual",
                lambda d: d["task"] == "mask_provided_facts")
uids = [u for u in pd.read_csv(C.RESULTS / "mask_factual_provided_conditions.csv").uid
        if u in ds.index]
print(f"{len(uids)} mask-factual first-run rows", flush=True)

existing = {f.split("/")[-1][:-3] for f in
            (s.rfilename for s in HfApi().repo_info(REPO, repo_type="dataset").siblings)
            if f.startswith("first_run/mask_factual/") and f.count("/") == 2
            and f.endswith(".pt") and "__" not in f}
print(f"{len(existing)} already on HF", flush=True)

tok = AutoTokenizer.from_pretrained(C.TARGET_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    C.TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager")
model.eval()


@torch.inference_mode()
def state(uid):
    r = ds.loc[uid]
    msgs = C.to_messages(r["trace"]) + [dict(r["lie_detection_prompt"])]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False)
    return torch.stack([h[0, -1, :] for h in out.hidden_states]).cpu()


# check against an original before doing the rest
ref = sorted(existing)[0]
old = torch.load(hf_hub_download(REPO, f"first_run/mask_factual/{ref}.pt", repo_type="dataset"))
new = state(ref)
diff = (new.float() - old.float()).abs()
rel = (diff.max() / old.float().abs().max()).item()
print(f"CHECK {ref}: shape {tuple(new.shape)} vs {tuple(old.shape)}, max abs diff "
      f"{diff.max().item():.4g}, relative to max |value| {rel:.2e}", flush=True)
if new.shape != old.shape or rel > 1e-2:
    raise SystemExit("CHECK FAILED - recapture does not match the original; nothing uploaded")

for uid in uids:
    s = state(uid)
    torch.save(s.to(torch.float16), F16 / f"{uid}.pt")
    torch.save(s.float(), F32 / f"{uid}.pt")
print(f"captured {len(list(F16.glob('*.pt')))} float16, {len(list(F32.glob('*.pt')))} float32",
      flush=True)

missing = STAGE / "fp16_missing"
missing.mkdir(exist_ok=True)
for f in F16.glob("*.pt"):
    if f.stem not in existing:
        (missing / f.name).write_bytes(f.read_bytes())
print(f"{len(list(missing.glob('*.pt')))} float16 files to add", flush=True)
upload(missing, "first_run/mask_factual", folder=True)
upload(F32, "first_run/mask_factual/fp32", folder=True)
print("08_mask_first_run_recapture: DONE", flush=True)
