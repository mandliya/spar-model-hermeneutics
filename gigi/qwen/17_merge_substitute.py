"""Merge every substitute box's results into results/<READER_TAG>_reads_<MODEL_TAG>/.

Each box (run_substitute.sh) uploads its results to
models/<MODEL_TAG>/substitute/<READER_TAG>/boxes/<JOB>/results/ on HF. This
downloads them all and combines files of the same name: JSONL lines are
concatenated, exact duplicates dropped. Nothing is generated, no API calls, no
GPU; runs anywhere.

A file that already exists locally is only replaced by a merged version that
contains every line it had (so nothing already here is lost).

Run from gigi/qwen/:  MODEL_TAG=qwen3-next-80b python 17_merge_substitute.py
"""
import collections
import json
import os
import pathlib

from huggingface_hub import HfApi, hf_hub_download

from src import common as C
from src.hf_sync import REPO, remote

READER_TAG = os.getenv("READER_TAG", "gemma-3-27b")
OUT = C.ROOT / "results" / f"{READER_TAG}_reads_{C.MODEL_TAG}"
OUT.mkdir(parents=True, exist_ok=True)
pfx = remote(f"substitute/{READER_TAG}/boxes")
api = HfApi(token=os.environ.get("HF_TOKEN"))
paths = [e.path for e in api.list_repo_tree(REPO, path_in_repo=pfx, repo_type="dataset", recursive=True)
         if "/results/" in getattr(e, "path", "") and hasattr(e, "size") and e.path.endswith(".jsonl")]
job = lambda rp: rp[len(pfx) + 1:].split("/")[0]
paths = [rp for rp in paths if job(rp) != "SMOKE"]   # a SMOKE box's rows are redone by the real jobs
files = collections.defaultdict(list)
for rp in sorted(paths):
    p = hf_hub_download(REPO, rp, repo_type="dataset", token=os.environ.get("HF_TOKEN"), force_download=True)
    files[pathlib.Path(rp).name].append((job(rp), pathlib.Path(p)))
print(f"{len(files)} distinct files from boxes {sorted({job(rp) for rp in paths})}")

for name, parts in sorted(files.items()):
    lines, seen = [], set()
    for _job, p in parts:
        for l in p.read_text().splitlines():
            if l.strip() and l not in seen:
                seen.add(l)
                lines.append(l)
    out = OUT / name
    if out.exists() and not set(l for l in out.read_text().splitlines() if l.strip()) <= seen:
        print(f"!! {name}: local file has lines the boxes don't - left alone")
        continue
    out.write_text("\n".join(lines) + "\n")
    recs = [json.loads(l) for l in lines]
    ok = {r["name"] for r in recs if "error" not in r}
    print(f"{name:34s} {'+'.join(j for j, _ in parts):12s} {len(ok):5d} answers, "
          f"{len(recs) - sum('error' not in r for r in recs)} error records, "
          f"{sum('error' not in r for r in recs) - len(ok)} duplicate answers")
