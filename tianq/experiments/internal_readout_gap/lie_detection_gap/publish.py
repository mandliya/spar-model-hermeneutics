"""Upload what gap.py, meaning_gap.py, judge_gap.py and rows.py made off the box to HF, then check it is there.

    cd tianq && python -m experiments.internal_readout_gap.lie_detection_gap.publish

gap.py runs where the arena's code runs, not on the box, so its outputs are not covered by shared/upload.py. This
sends scratch/internal-readout-gap/*gap_*.json (and the judge's cached answers, judge_gap/) to
internal-readout-gap/lie_detection_gap/outputs/ and
scratch/internal-readout-gap/logs/lie_detection_gap_* to internal-readout-gap/lie_detection_gap/logs/, then compares
sizes on the Hub. Never deletes anything on HF.
"""

from __future__ import annotations

import os
import pathlib

from huggingface_hub import HfApi

from dyl_repro.core.store import SCRATCH

LOCAL = pathlib.Path(SCRATCH) / "internal-readout-gap"
PREFIX = "internal-readout-gap/lie_detection_gap"


def main() -> None:
    api, repo = HfApi(token=os.environ["HF_TOKEN"]), os.environ["HF_DATASET_REPO"]
    files = {f"{PREFIX}/outputs/{p.name}": p for p in sorted(LOCAL.glob("*gap_*.json"))}
    files |= {f"{PREFIX}/outputs/judge_gap/{p.name}": p for p in sorted((LOCAL / "judge_gap").glob("*.jsonl"))}
    # this run's own log (lie_detection_gap_publish.log, when the output is saved there) is still being written
    files |= {f"{PREFIX}/logs/{p.name}": p for p in sorted((LOCAL / "logs").glob("lie_detection_gap_*"))
              if p.name != "lie_detection_gap_publish.log"}
    for remote, path in files.items():
        api.upload_file(path_or_fileobj=str(path), path_in_repo=remote, repo_id=repo, repo_type="dataset",
                        commit_message=f"internal readout gap: lie_detection_gap {path.name}")
    listed = {f.path: f.size for f in api.list_repo_tree(repo, path_in_repo=PREFIX, recursive=True, repo_type="dataset")
              if hasattr(f, "size")}
    missing = [r for r, p in files.items() if listed.get(r) != p.stat().st_size]
    print(f"published {len(files)} files; revision {api.list_repo_commits(repo, repo_type='dataset')[0].commit_id}", flush=True)
    print(f"completeness: {len(missing)} files missing or different" + (f", e.g. {missing[:5]}" if missing else ""), flush=True)
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
