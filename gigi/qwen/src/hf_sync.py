"""Copy results off the box as each stage finishes. Never raises: a failed
upload is logged loudly and the run carries on (the file is still on disk)."""

import os
import time

from huggingface_hub import HfApi

from src import common  # loads .env (HF_TOKEN); MODEL_TAG for the upload prefix

REPO = "model-hermeneutics/gigi-lie-detection-activations"


def remote(path_in_repo):
    """Every qwen upload lives under models/<MODEL_TAG>/ on HF, so it
    can never overwrite or move a gemma file."""
    pfx = f"models/{common.MODEL_TAG}/"
    return path_in_repo if path_in_repo.startswith(pfx) else pfx + path_in_repo


def upload(local, path_in_repo, folder=False):
    path_in_repo = remote(path_in_repo)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for attempt in range(5):
        try:
            if folder:
                api.upload_folder(folder_path=str(local), path_in_repo=path_in_repo,
                                  repo_id=REPO, repo_type="dataset",
                                  commit_message=f"Upload {path_in_repo}")
            else:
                api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                                repo_id=REPO, repo_type="dataset",
                                commit_message=f"Upload {path_in_repo}")
            print(f"UPLOADED {local} -> {path_in_repo}", flush=True)
            return True
        except Exception as e:
            print(f"upload attempt {attempt + 1} failed for {path_in_repo}: {e!r}"[:400], flush=True)
            time.sleep(30 * (attempt + 1))
    print(f"!!! UPLOAD FAILED for {path_in_repo} - file is still at {local}", flush=True)
    return False
