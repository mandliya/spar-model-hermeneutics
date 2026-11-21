"""Bring the earlier reads of the restart points back onto a box: saved activations, lens words and NLA sentences.

    python fetch_reads.py --models gemma-3-27b-it gemma-distilled-step96 --words_only qwen3.6-27b

Downloads internal-readout-gap/distillation_similarity/outputs/<model>/ from our HF dataset into
OUT/distillation_similarity/<model>/ (activations only for --models). Files already there are kept; rerun until it
reports 0 missing. Fails loudly if any file is still missing.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import models  # noqa: E402
from shared.upload import PREFIX, REPO  # noqa: E402

EXPERIMENT = "distillation_similarity"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="activations, lens words and NLA sentences")
    ap.add_argument("--words_only", nargs="*", default=[], help="lens words and NLA sentences only")
    args = ap.parse_args()
    token = os.environ["HF_TOKEN"]
    root = f"{PREFIX}/{EXPERIMENT}/outputs"
    tags = [*args.models, *args.words_only]
    patterns = [f"{root}/{m}/{kind}/*" for m in args.models for kind in ("acts", "lens", "nla")]
    patterns += [f"{root}/{m}/{kind}/*" for m in args.words_only for kind in ("lens", "nla")]
    patterns += [f"{root}/{m}/skipped.json" for m in tags]
    stage = models.OUT.parent / "fetched_reads"
    snapshot_download(REPO, repo_type="dataset", allow_patterns=patterns, local_dir=stage, max_workers=8, token=token)
    dest = models.OUT / EXPERIMENT
    dest.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)
    wanted = {f.path: f.size for m in tags
              for f in api.list_repo_tree(REPO, path_in_repo=f"{root}/{m}", recursive=True, repo_type="dataset")
              if hasattr(f, "size") and (m in args.models or "/acts/" not in f.path)}
    missing = 0
    for path, size in wanted.items():
        got = stage / path
        target = dest / pathlib.PurePosixPath(path).relative_to(root)
        if not target.exists() and got.exists() and got.stat().st_size == size:
            target.parent.mkdir(parents=True, exist_ok=True)
            got.rename(target)
        missing += not (target.exists() and target.stat().st_size == size)
    print(f"fetch_reads: {len(wanted)} files wanted, {missing} missing", flush=True)
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
