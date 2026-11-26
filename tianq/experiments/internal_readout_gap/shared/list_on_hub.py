"""What a folder of our HF dataset repo holds: file count and total bytes for it and for each folder one level down.

    python list_on_hub.py internal-readout-gap/classic_nla/outputs/saves/gemma-classic-step92

Reads only. For checking, from any machine with the token, that an upload is really there after its box is gone.
"""

from __future__ import annotations

import collections
import os
import pathlib
import sys

from huggingface_hub import HfApi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared.upload import REPO  # noqa: E402


def main() -> None:
    under = sys.argv[1].rstrip("/")
    files = [f for f in HfApi(token=os.environ["HF_TOKEN"]).list_repo_tree(REPO, path_in_repo=under, recursive=True, repo_type="dataset")
             if hasattr(f, "size")]
    by_folder: dict[str, list[int]] = collections.defaultdict(list)
    for f in files:
        below = f.path[len(under) + 1:].split("/")
        by_folder[below[0] if len(below) > 1 else "(files directly here)"].append(f.size)
    print(f"{REPO}/{under}: {len(files)} files, {sum(f.size for f in files):,} bytes")
    for folder, sizes in sorted(by_folder.items()):
        print(f"  {folder}: {len(sizes)} files, {sum(sizes):,} bytes")


if __name__ == "__main__":
    main()
