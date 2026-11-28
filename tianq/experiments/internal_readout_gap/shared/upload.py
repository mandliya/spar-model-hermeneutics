"""Upload one experiment's outputs and logs to HF, then check that every local file is there.

    python upload.py --experiment distillation_similarity

Uploads OUT/<experiment> to internal-readout-gap/<experiment>/outputs and that experiment's logs
(<experiment>_* and setup.log in /workspace/logs, or in READOUT_LOGS for an analysis run off the box on files from
fetch.py) to internal-readout-gap/<experiment>/logs in our dataset repo, then
lists both folders on the Hub and compares every local file's size (the completeness check before a box is
destroyed). The check leaves out this upload's own log, which is still being written. Never deletes anything on HF.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

from huggingface_hub import HfApi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import models  # noqa: E402

REPO = "model-hermeneutics/dyl-probe-repro"
PREFIX = "internal-readout-gap"
LOGS = pathlib.Path(os.environ.get("READOUT_LOGS", "/workspace/logs"))   # off the box: a folder holding only that run's logs
ATTEMPTS = 4   # a folder of thousands of files can lose one file to the network; files already sent are not sent again


def not_on_hub(api: HfApi, local: dict[str, int], under: str) -> list[str]:
    """The completeness check: of the files in `local` (repo path -> size in bytes), those that are missing under the
    repo folder `under` on the Hub or have another size there."""
    listed = {f.path: f.size for f in api.list_repo_tree(REPO, path_in_repo=under, recursive=True, repo_type="dataset")
              if hasattr(f, "size")}
    return [path for path, size in local.items() if listed.get(path) != size]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    args = ap.parse_args()
    e = args.experiment
    api = HfApi(token=os.environ["HF_TOKEN"])
    own_log = f"{e}_upload.log"
    # One commit per folder, its own files only: the Hub's upload links expire after 15 minutes, and a whole
    # experiment in one batch (36,000 files, 33 GB for lie_detection_gap) outlives them. Files already there are skipped.
    root = models.OUT / e
    for folder in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        if any(p.is_file() for p in folder.iterdir()):
            rel = folder.relative_to(root).as_posix()
            for attempt in range(1, ATTEMPTS + 1):
                try:
                    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(folder), ignore_patterns=["*/*"],
                                      path_in_repo=f"{PREFIX}/{e}/outputs" + ("" if rel == "." else f"/{rel}"),
                                      commit_message=f"internal readout gap: {e} outputs {rel}")
                    break
                except RuntimeError as error:   # huggingface_hub's "Error while uploading '<file>' to the Hub."
                    if attempt == ATTEMPTS:
                        raise
                    print(f"folder {rel}: attempt {attempt} failed ({error}); trying again in a minute", flush=True)
                    time.sleep(60)
            print(f"uploaded folder {rel}", flush=True)
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(LOGS), path_in_repo=f"{PREFIX}/{e}/logs",
                      allow_patterns=[f"{e}_*", "setup.log"], commit_message=f"internal readout gap: {e} logs")
    local = {f"{PREFIX}/{e}/outputs/{p.relative_to(models.OUT / e).as_posix()}": p.stat().st_size
             for p in (models.OUT / e).rglob("*") if p.is_file()}
    local |= {f"{PREFIX}/{e}/logs/{p.name}": p.stat().st_size for p in LOGS.iterdir()
              if p.is_file() and (p.name.startswith(f"{e}_") or p.name == "setup.log") and p.name != own_log}
    missing = not_on_hub(api, local, f"{PREFIX}/{e}")
    print(f"uploaded {e}: {len(local)} local files; revision {api.list_repo_commits(REPO, repo_type='dataset')[0].commit_id}",
          flush=True)
    print(f"completeness: {len(missing)} files missing or different" + (f", e.g. {missing[:5]}" if missing else ""), flush=True)
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
