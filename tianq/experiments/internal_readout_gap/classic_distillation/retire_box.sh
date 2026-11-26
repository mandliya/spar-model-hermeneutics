#!/usr/bin/env bash
# Run on a box that is no longer needed: send its logs to HF under names of their own, then destroy the box.
#   bash retire_box.sh <experiment> <label> "<reason>"
#   bash retire_box.sh classic_nla first_box "too little GPU memory for the 27B training"
# The logs /workspace/logs/<experiment>_* are copied to <experiment>_<label>_*, so a later box's logs of the same
# experiment do not replace them on HF, and uploaded with shared/upload.py (which checks every file's size).
# The box is destroyed only if that check passes. Nothing on HF is deleted.
set -euo pipefail
E=$1; LABEL=$2; REASON=$3
HERE=/workspace/repo/tianq/experiments/internal_readout_gap
set -a; . /workspace/.env; set +a
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1

KEEP=/workspace/logs_$LABEL
mkdir -p "$KEEP" "/workspace/out/internal-readout-gap/$E"
for f in /workspace/logs/"$E"_* /workspace/logs/setup.log; do
  [ -f "$f" ] || continue
  name=$(basename "$f")
  cp "$f" "$KEEP/${E}_${LABEL}_${name#"${E}"_}"
done
python3 "$HERE/shared/blank_keys.py" "$KEEP"   # the install log traced the keys file; HF refuses live tokens
ls -la "$KEEP"
READOUT_LOGS=$KEEP /workspace/venvs/readout/bin/python "$HERE/shared/upload.py" --experiment "$E"

# The proof that the check passed has to leave the box before the box does (on 25 Nov it did not: the check's own
# line and the destroy reply were written after the logs were copied, and went with the box). One small file, on its
# own. If this upload fails the box is still destroyed.
NOTE=$KEEP/${E}_${LABEL}_retired.txt
echo "$(date -u +%FT%TZ) $E $LABEL: the upload's size check passed with 0 files missing or different; this box (Vast instance $(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_ID=//p')) is destroyed next. Reason: $REASON" > "$NOTE"
NOTE=$NOTE E=$E /workspace/venvs/readout/bin/python -c '
import os
from huggingface_hub import HfApi
note = os.environ["NOTE"]
HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj=note, repo_id=os.environ["HF_DATASET_REPO"], repo_type="dataset",
    path_in_repo="internal-readout-gap/" + os.environ["E"] + "/logs/" + os.path.basename(note))
print("uploaded", os.path.basename(note))' || echo "the retired note did not upload"

bash "$HERE/own_lens/destroy_box.sh" now "$REASON"
