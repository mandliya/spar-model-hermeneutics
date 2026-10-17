#!/usr/bin/env bash
# float32 recapture of every follow-up activation, one category at a time:
# capture -> tar -> upload to HF -> verify -> clear local copy -> next.
# (~56 GB total won't fit beside the 52 GB model, so categories can't pile up.)
# Local files are cleared ONLY after the HF upload verifies. Ends by stopping
# the box with FORCE_STOP=1, overriding the hold that kept it up for this job.
cd /workspace/lies-experiment
source /venv/main/bin/activate
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
REPO=model-hermeneutics/gigi-lie-detection-activations

say "waiting for the alibi pipeline to finish"
while pgrep -f '^bash chain_alibi\.sh$' >/dev/null; do sleep 60; done
say "alibi done - extending hold to 22:15 UTC for this job"
date -u -d '22:15' +%s > HOLD_STOP_UNTIL

for CAT in mask_factual unanswerable sycophancy_a sycophancy_b ascii capture_secret alibi; do
  say "=== $CAT: capture ==="
  python 04_recapture_fp32.py --category $CAT || say "!!! $CAT capture failed (resumable)"
  DIR=/workspace/fp32_staging/$CAT
  N=$(ls $DIR 2>/dev/null | wc -l)
  if [ "$N" -lt 1 ]; then say "!!! $CAT: no tensors - skipping upload"; continue; fi
  TAR=/workspace/fp32_${CAT}.tar.gz
  say "$CAT: $N tensors - tarring"
  tar czf "$TAR" -C "$DIR" . || { say "!!! tar failed"; continue; }
  say "$CAT: uploading $(du -h $TAR | cut -f1)"
  if python - "$TAR" "$CAT" <<'PY'
import os, sys
from dotenv import load_dotenv
load_dotenv("/workspace/lies-experiment/.env")
from huggingface_hub import HfApi
tar, cat = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "model-hermeneutics/gigi-lie-detection-activations"
dest = f"tarballs/fp32/{cat}_fp32.tar.gz"
api.upload_file(path_or_fileobj=tar, path_in_repo=dest, repo_id=repo, repo_type="dataset")
api.upload_file(path_or_fileobj=f"results/fp32_map_{cat}.jsonl",
                path_in_repo=f"maps/fp32/fp32_map_{cat}.jsonl", repo_id=repo, repo_type="dataset")
size = next(s.size for s in api.dataset_info(repo, files_metadata=True).siblings if s.rfilename == dest)
assert size == os.path.getsize(tar), f"size mismatch {size} vs {os.path.getsize(tar)}"
print(f"VERIFIED {dest} {size/1e9:.2f} GB")
PY
  then
    say "$CAT: verified on HF - clearing local copy"
    rm -rf "$DIR" "$TAR"
  else
    say "!!! $CAT upload NOT verified - keeping local files, stopping the sequence"
    break
  fi
done

say "=== fp32 COMPLETE - stopping instance ==="
FORCE_STOP=1 bash stop_self.sh
