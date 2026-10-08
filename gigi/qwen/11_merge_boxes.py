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
from huggingface_hub import HfApi, hf_hub_download

from src import common as C
from src.hf_sync import REPO, remote

pfx = remote("boxes/")
# one listing of the boxes folder, then one small download per file (a bulk
# snapshot_download of this large repo hung for 35 min on 5 Oct)
api = HfApi(token=os.environ.get("HF_TOKEN"))
paths = [e.path for e in api.list_repo_tree(REPO, path_in_repo=pfx.rstrip("/"), repo_type="dataset", recursive=True)
         if "/results/" in getattr(e, "path", "") and hasattr(e, "size")]
# J6's mask-factual stopped after 34 rows (empty-trace crash); J6b redid it in full,
# so J6's partial mask-factual files are left out to avoid counting rows twice. Their
# lines still count as known below: an earlier merge left them in the local copies.
is_partial = lambda rp: rp.split("/")[3] == "J6" and "mask_factual" in rp
partial = {pathlib.Path(rp).name: rp for rp in paths if is_partial(rp)}
paths = [rp for rp in paths if not is_partial(rp)]
# The SMOKE box's 2-row runs are only for timing; the real jobs redo those rows.
paths = [rp for rp in paths if rp.split("/")[3] != "SMOKE"]
files = collections.defaultdict(list)
for i, rp in enumerate(sorted(paths)):
    p = pathlib.Path(hf_hub_download(REPO, rp, repo_type="dataset", token=os.environ.get("HF_TOKEN")))
    files[pathlib.Path(rp).name].append(p)
    if i % 25 == 0:
        print(f"downloaded {i + 1}/{len(paths)}", flush=True)
print(f"{len(files)} distinct files from {len({rp.split('/')[3] for rp in paths})} boxes")

for name, paths in sorted(files.items()):
    out = C.RESULTS / name
    if name.endswith(".jsonl"):
        lines, seen = [], set()
        for p in paths:
            for l in p.read_text().splitlines():
                if l.strip() and l not in seen:
                    seen.add(l); lines.append(l)
        known = seen | (set(pathlib.Path(hf_hub_download(REPO, partial[name], repo_type="dataset",
                                                          token=os.environ.get("HF_TOKEN"))).read_text().splitlines())
                        if name in partial else set())
        if out.exists() and not set(l for l in out.read_text().splitlines() if l.strip()) <= known:
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
