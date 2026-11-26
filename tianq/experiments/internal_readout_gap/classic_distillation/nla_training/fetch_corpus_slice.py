"""Download the slice of Ultra-FineWeb the activations are collected on.

    python fetch_corpus_slice.py --documents 100000 --to /workspace/nla/corpus

Why a slice: the library's own note (configs/datagen/qwen7b_ultrafineweb_100k.yaml) says the 1B-token Ultra-FineWeb
subset its released checkpoints were trained on is no longer hosted, that the full dataset is multi-TB and its data
stage would download all of it, and to "pre-download an Ultra-FineWeb slice locally and point corpus.name at the local
path". The slice taken here is the plainest one: the English split's part files in their own order, from the first,
until they hold the asked number of documents. Whole part files are kept; the data stage then reads documents
0 .. N-1 of them, as their config's corpus.start and corpus.length say. The dataset is pinned to one commit.

Prints each file with its row count, so the log says exactly which documents were used. Skips files already there.
"""

from __future__ import annotations

import argparse
import pathlib

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "openbmb/Ultra-FineWeb"
REVISION = "02c85641e3d19a854be2e09139c25adaa9518063"   # the dataset's head on 25 Nov 2026
FOLDER = "data/ultrafineweb_en"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--documents", type=int, required=True)
    ap.add_argument("--to", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.to)
    out.mkdir(parents=True, exist_ok=True)
    parts = sorted(f.path for f in HfApi().list_repo_tree(REPO, path_in_repo=FOLDER, repo_type="dataset", revision=REVISION)
                   if f.path.endswith(".parquet"))
    print(f"{REPO}@{REVISION}: {len(parts)} English part files", flush=True)
    have = 0
    for part in parts:
        local = out / pathlib.Path(part).name
        if not local.exists():
            fetched = hf_hub_download(REPO, part, repo_type="dataset", revision=REVISION, local_dir=out / "download")
            pathlib.Path(fetched).rename(local)
        rows = pq.ParquetFile(local).metadata.num_rows
        have += rows
        print(f"{local.name}: {rows:,} documents, {local.stat().st_size:,} bytes; {have:,} so far", flush=True)
        if have >= args.documents:
            break
    if have < args.documents:
        raise SystemExit(f"only {have:,} documents in the whole split, {args.documents:,} asked")
    print(f"corpus slice ready: {have:,} documents in {out}", flush=True)


if __name__ == "__main__":
    main()
