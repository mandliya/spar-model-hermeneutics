# ARCHIVED (8 Nov 2026 first pass). Kept only so that experiment can be reproduced; do not import or extend.
# For new work use ../shared/ (upload with a completeness check).
"""Upload this experiment's outputs and logs to HF, under internal-readout-gap/ in our dataset repo.

    python upload.py --part part1

Uploads OUT (activations, lens readouts, NLA explanations, comparisons, the fitted lenses) and /workspace/logs, then
lists the repo folder and checks every local file is there with the same size (the completeness check before a box is
destroyed). The check skips this upload's own log (part<N>_upload.log), which is still being written while it runs.
Never deletes anything on HF.
"""

from __future__ import annotations

import argparse
import os
import pathlib

from huggingface_hub import HfApi

import models

REPO = "model-hermeneutics/dyl-probe-repro"
PREFIX = "internal-readout-gap"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    args = ap.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])
    sources = {pathlib.Path(models.OUT): f"{PREFIX}/outputs", pathlib.Path("/workspace/logs"): f"{PREFIX}/logs/{args.part}"}
    for local, remote in sources.items():
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(local), path_in_repo=remote,
                          ignore_patterns=["*_fit_checkpoint.pt"], commit_message=f"internal readout gap: {args.part} {remote}")
        print(f"uploaded {local} -> {REPO}/{remote}", flush=True)
    missing = []
    for local, remote in sources.items():
        listed = {f.path: f.size for f in api.list_repo_tree(REPO, path_in_repo=remote, recursive=True, repo_type="dataset")
                  if hasattr(f, "size")}
        for p in local.rglob("*"):
            if p.is_file() and not p.name.endswith("_fit_checkpoint.pt") and p.name != f"{args.part}_upload.log":
                key = f"{remote}/{p.relative_to(local).as_posix()}"
                if listed.get(key) != p.stat().st_size:
                    missing.append(key)
    print(f"completeness: {len(missing)} files missing or different" + (f", e.g. {missing[:5]}" if missing else ""), flush=True)
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
