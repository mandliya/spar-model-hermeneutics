"""Upload one folder of an experiment's outputs ahead of the full upload, for small files an analysis is waiting on.

    python upload_one_folder.py --experiment gender_secret_rerun --folder distributions

shared/upload.py sends an experiment folder by folder in name order, and a folder of thousands of activation files
takes a quarter of an hour. This sends OUT/<experiment>/<folder> (its own files only) to the same place on the Hub
now. The full upload later finds those files already there and its size check still covers them.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

from huggingface_hub import HfApi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import models, upload  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--folder", required=True)
    args = ap.parse_args()
    folder = models.OUT / args.experiment / args.folder
    under = f"{upload.PREFIX}/{args.experiment}/outputs/{args.folder}"
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_folder(repo_id=upload.REPO, repo_type="dataset", folder_path=str(folder), ignore_patterns=["*/*"],
                      path_in_repo=under, commit_message=f"internal readout gap: {args.experiment} outputs {args.folder}")
    local = {f"{under}/{p.name}": p.stat().st_size for p in folder.iterdir() if p.is_file()}
    missing = upload.not_on_hub(api, local, under)
    print(f"uploaded {args.folder}: {len(local)} files, {len(missing)} missing or different", flush=True)
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
