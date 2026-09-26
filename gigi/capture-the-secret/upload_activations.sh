#!/usr/bin/env bash
# Tar and upload activation tensors to the private HF dataset repo.
#
# Tarballs, not loose files: HF caps commits at 256/hour and this project has
# ~16k tensors, which as individual uploads would need ~800 commits. Two
# tarballs is two commits.
#
# Takes a name prefix to select which files to bundle, so a category can be
# backed up as soon as its run finishes rather than waiting for all of them.
#
#   bash upload_activations.sh unanswerable 'unanswerable_*'
#
# Network-bound, so it does not compete with a GPU job running alongside it.

set -o pipefail
cd "$(dirname "$0")"
LABEL="$1"; PATTERN="$2"
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }

if [ -z "$LABEL" ] || [ -z "$PATTERN" ]; then
  say "usage: bash upload_activations.sh <label> <filename-glob>"; exit 1
fi

cd results/activations || { say "!!! no results/activations"; exit 1; }
N=$(ls $PATTERN 2>/dev/null | wc -l | tr -d ' ')
say "matched $N files for '$PATTERN'"
if [ "$N" -lt 1 ]; then
  say "!!! ABORT: nothing matched — refusing to upload an empty tarball"
  exit 1
fi

TAR="/workspace/${LABEL}_activations.tar.gz"
say "creating $TAR ..."
ls $PATTERN | tar czf "$TAR" -T - || { say "!!! tar failed"; exit 1; }
SZ=$(du -h "$TAR" | cut -f1)
say "tarball size: $SZ"

# Refuse to upload something implausibly small (a previous run once shipped
# two 45-byte tarballs built from empty globs).
BYTES=$(stat -c %s "$TAR")
if [ "$BYTES" -lt 100000 ]; then
  say "!!! ABORT: tarball is only $BYTES bytes — that cannot be $N tensors"
  exit 1
fi

cd /workspace/lies-experiment
python - "$TAR" "$LABEL" <<'PY'
import os, sys
from dotenv import load_dotenv
load_dotenv("/workspace/lies-experiment/.env")
from huggingface_hub import HfApi
tar, label = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "model-hermeneutics/gigi-lie-detection-activations"
dest = f"tarballs/{label}_activations.tar.gz"
print(f"uploading {tar} -> {repo}:{dest}", flush=True)
api.upload_file(path_or_fileobj=tar, path_in_repo=dest, repo_id=repo,
                repo_type="dataset")
info = api.dataset_info(repo, files_metadata=True)
for s in info.siblings or []:
    if s.rfilename == dest:
        print(f"VERIFIED {dest}  {s.size/1e9:.2f} GB")
PY
say "=== $LABEL upload COMPLETE ==="
