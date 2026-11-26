"""On a fresh box, fetch from HF what continue_training.sh needs to carry on from a save.

    /workspace/venvs/readout/bin/python restore_from_hf.py --tag gemma-classic-step92 --save iter_0000005

Fetches, from internal-readout-gap/classic_nla/outputs/ in our dataset repo:
  saves/<tag>/<save>/run/     -> /workspace/nla/runs/<tag>/         both models' training saves, the round counters and
                                                                    the data position, laid out as the trainer left them
  data/<tag>/rl_shuf.parquet* -> /workspace/out/internal-readout-gap/classic_nla/data/<tag>/   the training activations
The released pair comes from prepare_models.sh as on the first box. Skips files already there. Then
`bash run.sh train` with ROUNDS set carries on from the save's round.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

from huggingface_hub import snapshot_download

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from shared import models  # noqa: E402
from shared.upload import PREFIX, REPO  # noqa: E402

EXPERIMENT = "classic_nla"
RUNS = pathlib.Path("/workspace/nla/runs")


def fetch(pattern: str, under: str, to: pathlib.Path) -> None:
    """The repo files matching `pattern` below the repo folder `under`, placed in `to` with their paths below `under`."""
    got = pathlib.Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"{under}/{pattern}"], token=os.environ["HF_TOKEN"]))
    for f in sorted((got / under).rglob("*")):
        if f.is_file():
            target = to / f.relative_to(got / under)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy(f, target)   # out of the HF cache: the trainer writes into this folder
            print(f"{target}: {target.stat().st_size:,} bytes", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save", required=True, help="the save's folder name on HF, e.g. iter_0000005")
    args = ap.parse_args()
    outputs = f"{PREFIX}/{EXPERIMENT}/outputs"
    fetch("**", f"{outputs}/saves/{args.tag}/{args.save}/run", RUNS / args.tag)
    fetch("rl_shuf.parquet*", f"{outputs}/data/{args.tag}", models.OUT / EXPERIMENT / "data" / args.tag)
    print(f"restored {args.save}: newest writer save is round {(RUNS / args.tag / 'actor' / 'latest_checkpointed_iteration.txt').read_text()}")


if __name__ == "__main__":
    main()
