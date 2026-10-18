"""Upload the raw forensics runs (every step's transcript and state) to the project's HF dataset.

One tarball per stage: results/raw/<stage>/ -> forensics/raw/<stage>.tar.gz in
model-hermeneutics/gigi-lie-detection-activations. Uploading a stage again replaces the file
(HF keeps the history). A stage is refused if any file in it holds an API key (some
environments, e.g. swe_rebench_v2, let the agent's shell read the key, so a transcript can carry it).

Run:  python gigi/3-forensics/scripts/upload_raw.py [stage ...]     (needs HF_TOKEN; default: all stages)
"""
import os
import pathlib
import re
import sys
import tarfile
import tempfile

from huggingface_hub import HfApi

HERE = pathlib.Path(__file__).resolve().parents[1]
RAW = HERE / "results" / "raw"
REPO = "model-hermeneutics/gigi-lie-detection-activations"


KEY_PATTERNS = [re.compile(rb"sk-or-v1-[0-9a-f]{20,}"), re.compile(rb"sk-ant-[A-Za-z0-9_-]{20,}"), re.compile(rb"hf_[A-Za-z0-9]{30,}")]


def leaks(src):
    """Files under src that hold anything shaped like an API key, or the keys in this environment."""
    mine = [v.encode() for k, v in os.environ.items() if k.endswith(("_API_KEY", "_TOKEN")) and len(v) > 15]
    bad = []
    for f in src.rglob("*"):
        if f.is_file():
            b = f.read_bytes()
            if any(m in b for m in mine) or any(p.search(b) for p in KEY_PATTERNS):
                bad.append(f)
    return bad


def main(stages):
    api = HfApi(token=os.environ["HF_TOKEN"])
    stages = stages or sorted(p.name for p in RAW.iterdir() if p.is_dir())
    for stage in stages:
        src = RAW / stage
        bad = leaks(src)
        if bad:
            print(f"REFUSED {stage}: {len(bad)} file(s) hold an API key, e.g. {bad[0]}; not uploaded")
            continue
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
