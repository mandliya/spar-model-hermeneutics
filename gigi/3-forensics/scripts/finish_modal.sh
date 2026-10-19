#!/usr/bin/env bash
# After a Modal pass: wait until none of our apps is running (8 h at most), mirror the volumes to
# results/raw/modal/, then upload the raw stages to the HF dataset (upload_raw.py refuses any stage
# holding an API key). Logs to results/logs/<stage>/finish.txt. Safe to rerun: sync and upload are idempotent.
#   MODAL_PROFILE=... MODAL_VOLUME_PREFIX=... HF_TOKEN=... bash gigi/3-forensics/scripts/finish_modal.sh <stage> [upload stage ...]
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; STAGE=$1; shift; LOG="$HERE/results/logs/$STAGE"; mkdir -p "$LOG"
say () { echo "$(date '+%F %T') $*" >> "$LOG/finish.txt"; }
cd "$HERE/agent-interp-envs"
running () { env -u VIRTUAL_ENV uv run modal app list --json 2>/dev/null | python3 -c "
import sys, json   # ours only: the workspace may be shared
print(sum('stopped' not in str(a.get('state', '')).lower() and '$HERE/configs/' in json.dumps(a) for a in json.load(sys.stdin)))"; }
for i in $(seq 1 96); do n=$(running); say "our apps running: ${n:-?}"; [ "$n" = 0 ] && break; sleep 300; done
say "syncing"; env -u VIRTUAL_ENV uv run python ../scripts/modal_sync.py >> "$LOG/finish_sync.log" 2>&1 || say "sync exited non-zero"
say "uploading: $*"; env -u VIRTUAL_ENV uv run python ../scripts/upload_raw.py "$@" >> "$LOG/finish_upload.log" 2>&1 && say "uploaded" || say "upload exited non-zero"
say "done"
