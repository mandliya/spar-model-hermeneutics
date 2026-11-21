"""Upload one finished lens fit to HF at once, before the run's full upload, so a lost box cannot lose it.

    python upload_lens.py --fit gemma-3-27b-it

Uploads OUT/own_lens/<fit>/ (the final lens, its convergence log, its snapshots) to the place shared/upload.py
puts it, internal-readout-gap/own_lens/outputs/<fit>/, so the later full upload finds the files there and skips
them. The resumable checkpoint of a fit that is still running is left out. Then checks every uploaded file's size.
Never deletes anything on HF.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

from huggingface_hub import HfApi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import models  # noqa: E402
from shared.upload import PREFIX, REPO  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    args = ap.parse_args()
    folder = models.OUT / "own_lens" / args.fit
    if not list(folder.glob("*_jacobian_lens.pt")):
        raise SystemExit(f"{folder} has no final lens yet")
    api = HfApi(token=os.environ["HF_TOKEN"])
    dest = f"{PREFIX}/own_lens/outputs/{args.fit}"
    for sub in [folder, *sorted(p for p in folder.rglob("*") if p.is_dir())]:
        rel = sub.relative_to(folder).as_posix()
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(sub), ignore_patterns=["*/*", "*_checkpoint.pt", "*.partial"],
                          path_in_repo=dest + ("" if rel == "." else f"/{rel}"),
                          commit_message=f"internal readout gap: own_lens {args.fit} {rel}")
        print(f"uploaded folder {rel}", flush=True)
    local = {f"{dest}/{p.relative_to(folder).as_posix()}": p.stat().st_size for p in folder.rglob("*")
             if p.is_file() and not p.name.endswith(("_checkpoint.pt", ".partial"))}
    listed = {f.path: f.size for f in api.list_repo_tree(REPO, path_in_repo=dest, recursive=True, repo_type="dataset") if hasattr(f, "size")}
    missing = [k for k, size in local.items() if listed.get(k) != size]
    print(f"uploaded {args.fit}: {len(local)} files; revision {api.list_repo_commits(REPO, repo_type='dataset')[0].commit_id}", flush=True)
    print(f"completeness: {len(missing)} files missing or different" + (f", e.g. {missing[:5]}" if missing else ""), flush=True)
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
