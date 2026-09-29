#!/usr/bin/env bash
# One box session: named_cot for all seven categories, then the two float32
# categories that didn't finish overnight. Every category is backed up to HF
# the moment it finishes (tensors + results), and its local tensors are
# cleared only after the upload verifies by size. No `set -e`: the final stop
# always runs. Stopping preserves /workspace, so it is always safe.
cd /workspace/lies-experiment
source /venv/main/bin/activate
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
REPO=model-hermeneutics/gigi-lie-detection-activations
date -u -d '+10 hours' +%s > HOLD_STOP_UNTIL     # keep the box up for this job only

upload () {   # upload <staging-dir> <tar-name> <results-file> <hf-folder-for-results>
  local DIR=$1 NAME=$2 RES=$3 DEST=$4
  local N=$(ls "$DIR" 2>/dev/null | wc -l)
  if [ "$N" -lt 1 ]; then say "!!! $NAME: no tensors - nothing to upload"; return 1; fi
  local TAR=/workspace/${NAME}.tar.gz
  tar czf "$TAR" -C "$DIR" . || { say "!!! $NAME: tar failed"; return 1; }
  say "$NAME: uploading $N tensors ($(du -h $TAR | cut -f1))"
  if python - "$TAR" "$NAME" "$RES" "$DEST" <<'PY'
import os, sys
from dotenv import load_dotenv
load_dotenv("/workspace/lies-experiment/.env")
from huggingface_hub import HfApi
tar, name, res, folder = sys.argv[1:5]
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "model-hermeneutics/gigi-lie-detection-activations"
dest = f"tarballs/fp32/{name}.tar.gz"
api.upload_file(path_or_fileobj=tar, path_in_repo=dest, repo_id=repo, repo_type="dataset")
if res and os.path.exists(res):
    api.upload_file(path_or_fileobj=res, path_in_repo=f"{folder}/{os.path.basename(res)}",
                    repo_id=repo, repo_type="dataset")
    print(f"VERIFIED {folder}/{os.path.basename(res)}")
size = next(s.size for s in api.dataset_info(repo, files_metadata=True).siblings if s.rfilename == dest)
assert size == os.path.getsize(tar), f"size mismatch {size} vs {os.path.getsize(tar)}"
print(f"VERIFIED {dest} {size/1e9:.2f} GB")
PY
  then rm -rf "$DIR" "$TAR"; say "$NAME: verified on HF, local copy cleared"
  else say "!!! $NAME: upload NOT verified - keeping local files"; return 1; fi
}

say "disk before start: $(df -h /workspace | tail -1 | awk '{print $4}') free"

for CAT in alibi capture_secret ascii sycophancy_b sycophancy_a unanswerable mask_factual; do
  say "=== named_cot: $CAT ==="
  python 05_named_cot.py --category $CAT || say "!!! named_cot $CAT failed (resumable)"
  upload /workspace/fp32_staging/named_cot_$CAT named_cot_${CAT}_fp32 results/named_cot_$CAT.jsonl results/named_cot
done

for CAT in capture_secret alibi; do
  say "=== float32 recapture: $CAT ==="
  python 04_recapture_fp32.py --category $CAT || say "!!! fp32 $CAT failed (resumable)"
  upload /workspace/fp32_staging/$CAT ${CAT}_fp32 results/fp32_map_$CAT.jsonl maps/fp32
done

say "=== SESSION COMPLETE - stopping instance ==="
FORCE_STOP=1 bash stop_self.sh
