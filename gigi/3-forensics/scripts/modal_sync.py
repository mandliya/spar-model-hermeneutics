"""Mirror the forensics runs from Modal's per-environment volumes to results/raw/modal/<volume>/.

run_modal.py only downloads rollout.log at the end of a launch, and only if the laptop is still
connected; the step checkpoints (state.json, messages.json, fs/) stay on the volume. Each volume is
fetched whole with `modal volume get` into a scratch folder (fast; file-by-file reads took minutes
per volume), then every file not here yet is moved in. Rerun at any time; never rewrites a file. Which launch made which run: results/logs/<stage>/<config>.log names the
run directory (/mnt/results/<env>/.../<timestamp>/run-N).

Run:  cd gigi/3-forensics/agent-interp-envs && uv run python ../scripts/modal_sync.py [volume ...]
      (default: every volume named $MODAL_VOLUME_PREFIX*, default results-*)
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

import os

import modal

HERE = pathlib.Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "raw" / "modal"


def volumes():
    return sorted(v.name for v in modal.Volume.objects.list() if v.name.startswith(os.environ.get("MODAL_VOLUME_PREFIX", "results-")))


def sync(name):
    new = skipped = 0
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(["modal", "volume", "get", name, "/", f"{td}/v"], capture_output=True, text=True)
        if r.returncode:
            print(f"{name}: download failed: {(r.stderr or r.stdout).strip()[-300:]}")
            return
        root = pathlib.Path(td) / "v"
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            dest = OUT / name / f.relative_to(root)
            if dest.exists():
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(f, dest)
            new += 1
    print(f"{name}: {new} new file(s), {skipped} already here")


if __name__ == "__main__":
    for v in sys.argv[1:] or volumes():
        sync(v)
