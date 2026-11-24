"""Fetch the own-lens run's small results from HF for analysis off the box.

    HF_TOKEN=... python fetch_results.py --to <folder>

Downloads from internal-readout-gap/ in our dataset repo into <folder>:
  distillation_similarity/   the similarity_*.json, where_it_moved_*.json and rate_gap_*.json results, distributions/
  distillation_positions/    positions_similarity.json
  lie_detection_positions/   each read's words.json.gz and probe_layer.npz (the bundles positions_gap.py reads)
  own_lens/                  everything but the lens files (.pt): convergence logs, compare_lenses.jsonl, notes, status
  */logs/                    every log
Files already there with the right size are kept. Fails loudly if anything is missing afterwards.
"""

from __future__ import annotations

import argparse
import re
import os
import pathlib
import shutil
import sys

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared.upload import PREFIX, REPO  # noqa: E402

WANTED = ["distillation_similarity/outputs/*.json", "distillation_similarity/outputs/distributions/*",
          "distillation_positions/outputs/*.json", "lie_detection_positions/outputs/*/words.json.gz",
          "lie_detection_positions/outputs/*/probe_layer.npz", "own_lens/outputs/*", "own_lens/outputs/notes/*",
          "own_lens/outputs/*/*.csv", "*/logs/*"]
EXPERIMENTS = ("own_lens", "distillation_similarity", "distillation_positions", "lie_detection_positions")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", required=True)
    args = ap.parse_args()
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    dest = pathlib.Path(args.to)
    files = {}
    for e in EXPERIMENTS:
        for f in api.list_repo_tree(REPO, path_in_repo=f"{PREFIX}/{e}", recursive=True, repo_type="dataset"):
            rel = f.path.removeprefix(PREFIX + "/")
            if hasattr(f, "size") and not rel.endswith(".pt") and any(re.fullmatch(re.escape(w).replace(r"\*", "[^/]*"), rel) for w in WANTED):   # a * stops at a slash
                files[rel] = f.size
    missing = 0
    for n, (rel, size) in enumerate(sorted(files.items()), 1):
        target = dest / rel
        if not (target.exists() and target.stat().st_size == size):
            got = hf_hub_download(REPO, f"{PREFIX}/{rel}", repo_type="dataset", token=token)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(got, target)
        missing += not (target.exists() and target.stat().st_size == size)
        if n % 50 == 0:
            print(f"  {n} of {len(files)}", flush=True)
    print(f"fetch_results: {len(files)} files, {sum(files.values()) / 1e9:.2f} GB, {missing} missing; in {dest}", flush=True)
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
