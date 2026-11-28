#!/usr/bin/env bash
# Download one HF model repo with the chunked (xet) backend, for a repo that is one very large file.
#   bash fetch_in_parallel.sh ceselder/qwen3.6-27b-nla-av
# box_lib.sh turns that backend off because it once stalled with a few files left, and plain HTTP takes many files at
# once. But plain HTTP gives one connection per file: a single 54 GB file came down at 19 MB/s on 26 Nov. This is the
# other choice for such a repo. Log: /workspace/logs/setup_fetch_<repo name>.log; its last line is the folder.
set -euo pipefail
set -a; . /workspace/.env; set +a
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=0 HF_XET_HIGH_PERFORMANCE=1
exec >> "/workspace/logs/setup_fetch_${1##*/}.log" 2>&1
echo "### $(date -u +%FT%TZ) fetch $1 with the chunked backend"
/workspace/venvs/readout/bin/python -c "
import sys, hf_xet
from huggingface_hub import snapshot_download
print('downloaded to', snapshot_download(sys.argv[1], max_workers=16), flush=True)" "$1"
echo "### $(date -u +%FT%TZ) done"
