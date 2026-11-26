"""Push a running lens fit's snapshots and resume copies to HF as they appear, so a lost box loses little.

    python push_fit_progress.py --fit gemma-classic-step92

Every minute: uploads each finished file under OUT/own_lens/<fit>/snapshots and OUT/own_lens/<fit>/resume that HF does
not have yet, into the folder upload_lens.py uses (internal-readout-gap/own_lens/outputs/<fit>/). A file is uploaded
once and never replaced. Stops once the fit has written its final lens and nothing is left to push.

To carry a fit on from HF on a new box: download the newest resume/checkpoint_at_<n>.pt, save it as
<fit_dir>/<slug>_checkpoint.pt, and start fit.sh again; fit_lens.py picks it up.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import models  # noqa: E402
from shared.upload import PREFIX, REPO  # noqa: E402

PUSHED_FOLDERS = ("snapshots", "resume")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    args = ap.parse_args()
    folder = models.OUT / "own_lens" / args.fit
    dest = f"{PREFIX}/own_lens/outputs/{args.fit}"
    api = HfApi(token=os.environ["HF_TOKEN"])
    try:
        on_hub = {f.path for f in api.list_repo_tree(REPO, path_in_repo=dest, recursive=True, repo_type="dataset") if hasattr(f, "size")}
    except EntryNotFoundError:   # nothing of this fit is on HF yet
        on_hub = set()
    print(f"{len(on_hub)} files of {args.fit} are already on HF", flush=True)
    while True:
        finished = any(folder.glob("*_jacobian_lens.pt"))
        waiting = [p for sub in PUSHED_FOLDERS for p in sorted((folder / sub).glob("*.pt"))
                   if f"{dest}/{p.relative_to(folder).as_posix()}" not in on_hub]
        for p in waiting:
            target = f"{dest}/{p.relative_to(folder).as_posix()}"
            api.upload_file(path_or_fileobj=str(p), path_in_repo=target, repo_id=REPO, repo_type="dataset",
                            commit_message=f"internal readout gap: own_lens {args.fit} {p.relative_to(folder).as_posix()}")
            on_hub.add(target)
            print(f"{time.strftime('%FT%TZ', time.gmtime())} pushed {target} ({p.stat().st_size:,} bytes)", flush=True)
        if finished and not waiting:
            print("the fit has written its final lens and everything is pushed; stopping", flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
