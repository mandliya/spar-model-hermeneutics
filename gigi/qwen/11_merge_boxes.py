"""Merge every box's results into results/<MODEL_TAG>/ for the analysis.

Each box (run_job.sh) uploads its results to models/<MODEL_TAG>/boxes/<JOB>/
results/ on HF. This downloads them all and combines files of the same name:
JSONL lines are concatenated (exact duplicates dropped, e.g. the shared input
files), CSVs are concatenated and de-duplicated. Nothing is generated, no API
calls, no GPU; runs anywhere (laptop or cloud session).

A file that already exists locally is only replaced by a merged version that
contains every line it had (so nothing already here is lost).

Run:  python 11_merge_boxes.py
"""
import collections
import io
import os
import pathlib

import pandas as pd
from huggingface_hub import snapshot_download

from src import common as C
from src.hf_sync import REPO, remote

pfx = remote("boxes/")
local = pathlib.Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=[pfx + "*/results/*"],
                                       token=os.environ.get("HF_TOKEN")))
files = collections.defaultdict(list)
for p in sorted((local / pfx).glob("*/results/*")):
    files[p.name].append(p)
print(f"{len(files)} distinct files from {len({p.parent.parent.name for v in files.values() for p in v})} boxes")

for name, paths in sorted(files.items()):
    out = C.RESULTS / name
    if name.endswith(".jsonl"):
        lines, seen = [], set()
        for p in paths:
            for l in p.read_text().splitlines():
                if l.strip() and l not in seen:
                    seen.add(l); lines.append(l)
        if out.exists() and not set(l for l in out.read_text().splitlines() if l.strip()) <= seen:
            print(f"!! {name}: local file has lines the boxes don't - left alone"); continue
        out.write_text("\n".join(lines) + "\n")
    elif name.endswith(".csv"):
        df = pd.concat([pd.read_csv(p, low_memory=False) for p in paths]).drop_duplicates()
        df.to_csv(out, index=False)
        lines = df
    else:
        out.write_bytes(paths[0].read_bytes())
        lines = [0]
    print(f"{name:55s} {len(paths)} box(es) -> {len(lines)} rows")
