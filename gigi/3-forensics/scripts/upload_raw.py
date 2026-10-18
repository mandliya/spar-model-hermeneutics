"""Upload the raw forensics runs (every step's transcript and state) to the project's HF dataset.

One tarball per stage: results/raw/<stage>/ -> forensics/raw/<stage>.tar.gz in
model-hermeneutics/gigi-lie-detection-activations. Uploading a stage again replaces the file
(HF keeps the history).

Run:  python gigi/3-forensics/scripts/upload_raw.py [stage ...]     (needs HF_TOKEN; default: all stages)
"""
import os
import pathlib
import sys
import tarfile
import tempfile

from huggingface_hub import HfApi

HERE = pathlib.Path(__file__).resolve().parents[1]
RAW = HERE / "results" / "raw"
REPO = "model-hermeneutics/gigi-lie-detection-activations"


def main(stages):
    api = HfApi(token=os.environ["HF_TOKEN"])
    stages = stages or sorted(p.name for p in RAW.iterdir() if p.is_dir())
    for stage in stages:
        src = RAW / stage
        with tempfile.TemporaryDirectory() as td:
            tb = pathlib.Path(td) / f"{stage}.tar.gz"
            with tarfile.open(tb, "w:gz") as t:
                t.add(src, arcname=stage)
            size = tb.stat().st_size
            api.upload_file(path_or_fileobj=str(tb), path_in_repo=f"forensics/raw/{stage}.tar.gz", repo_id=REPO,
                            repo_type="dataset", commit_message=f"forensics raw runs: {stage}")
            print(f"uploaded {stage}: {size / 1e6:.1f} MB -> forensics/raw/{stage}.tar.gz")


if __name__ == "__main__":
    main(sys.argv[1:])
