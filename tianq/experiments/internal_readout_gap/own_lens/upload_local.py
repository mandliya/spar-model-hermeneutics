"""Upload the analyses that ran off the box (logs, result files, figures, the report) to HF, and check they are there.

    HF_TOKEN=... python upload_local.py --folder <folder> [--to own_lens/local_analysis]

Uploads every file under <folder> to internal-readout-gap/<to>/ in our dataset repo, keeping the folder's layout,
then lists that place on the Hub and compares every local file's size. Prints the revision. Never deletes anything.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

from huggingface_hub import HfApi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared.upload import PREFIX, REPO  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--to", default="own_lens/local_analysis")
    args = ap.parse_args()
    folder = pathlib.Path(args.folder)
    dest = f"{PREFIX}/{args.to}"
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(folder), path_in_repo=dest,
                      commit_message=f"internal readout gap: {args.to}")
    local = {f"{dest}/{p.relative_to(folder).as_posix()}": p.stat().st_size for p in folder.rglob("*") if p.is_file()}
    listed = {f.path: f.size for f in api.list_repo_tree(REPO, path_in_repo=dest, recursive=True, repo_type="dataset") if hasattr(f, "size")}
    missing = [k for k, size in local.items() if listed.get(k) != size]
    print(f"uploaded {len(local)} files to {dest}; revision {api.list_repo_commits(REPO, repo_type='dataset')[0].commit_id}", flush=True)
    print(f"completeness: {len(missing)} files missing or different" + (f", e.g. {missing[:5]}" if missing else ""), flush=True)
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
