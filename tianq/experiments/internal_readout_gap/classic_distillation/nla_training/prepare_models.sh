#!/usr/bin/env bash
# The models the NLA run needs, on the NLA box. Idempotent; safe to rerun.
#   bash prepare_models.sh gemma-classic-step92
# - base gemma and Gigi's adapter, merged into /workspace/models/<tag> by own_lens/merge.py: the same merge, in the
#   same environment, as the lens box, so the lens and the NLA belong to one model
# - the released Gemma-27B NLA pair, the starting point of the continued training
# The commit of each download is printed, so the log says exactly which weights were used.
set -euo pipefail
TAG=$1
. "$(dirname "$0")/../../shared/box_lib.sh"
HERE=$REPO/tianq/experiments/internal_readout_gap

readout_venv
fetch_distilled "$TAG"
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" "$HERE/own_lens/merge.py" --model "$TAG"

# the released pair, as shared/models.py names it for base gemma
read -r WRITER RECONSTRUCTOR < <("$VENV/bin/python" -c "
import sys
sys.path.insert(0, '$HERE')
from shared import models
kind, writer, reconstructor = models.MODELS['$TAG'].nla
assert kind == 'official', kind
print(writer, reconstructor)")
dl "$WRITER" - model &
dl "$RECONSTRUCTOR" - model &
wait
for repo in "$WRITER" "$RECONSTRUCTOR"; do
  echo "$repo is at commit $(ls "$HF_HOME/hub/models--${repo//\//--}/snapshots")"
done
du -sh "/workspace/models/$TAG"
echo "models ready"
