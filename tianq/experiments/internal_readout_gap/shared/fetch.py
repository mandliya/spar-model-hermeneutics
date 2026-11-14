"""Fetch an experiment's small output files (no activations) from HF, for analyses that run off the box.

    python fetch.py --experiment distillation_similarity --to <folder>

Downloads the lens and NLA JSON files of every read, and the experiment's top-level JSON results, into
<folder>/<experiment>/, the layout the box has under OUT. Point READOUT_OUT at <folder> to run an analysis on it.
The files are thousands of small ones and the Hub rate-limits bursts, so this goes one file at a time and retries; files
already there are kept, so it can be rerun until it reports nothing missing.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time

from huggingface_hub import HfApi, hf_hub_download

REPO = "model-hermeneutics/dyl-probe-repro"
PREFIX = "internal-readout-gap"
KINDS = ("lens", "nla")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--to", required=True)
    args = ap.parse_args()
    token = os.environ["HF_TOKEN"]
    root = f"{PREFIX}/{args.experiment}/outputs"
    listed = [f.path for f in HfApi(token=token).list_repo_tree(REPO, path_in_repo=root, recursive=True, repo_type="dataset")
              if hasattr(f, "size") and f.path.endswith(".json")
              and (f.path.count("/") == root.count("/") + 1 or f.path.split("/")[-2] in KINDS)]
    dest = pathlib.Path(args.to) / args.experiment
    todo = [p for p in listed if not (dest / pathlib.PurePosixPath(p).relative_to(root)).exists()]
    print(f"fetch {args.experiment}: {len(listed)} files on the Hub, {len(todo)} to download", flush=True)
    failed = []
    for n, path in enumerate(todo, 1):
        target = dest / pathlib.PurePosixPath(path).relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            try:
                got = hf_hub_download(REPO, path, repo_type="dataset", token=token)
                target.write_bytes(pathlib.Path(got).read_bytes())
                break
            except Exception as e:   # a rate limit or a dropped connection: wait and try again
                wait = 20 * (attempt + 1)
                print(f"  {path}: {type(e).__name__}, waiting {wait} s", flush=True)
                time.sleep(wait)
        else:
            failed.append(path)
        if n % 250 == 0:
            print(f"  {n} of {len(todo)}", flush=True)
    print(f"fetch {args.experiment}: {len(todo) - len(failed)} downloaded, {len(failed)} still missing", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
