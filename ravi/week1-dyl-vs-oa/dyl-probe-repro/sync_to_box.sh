#!/usr/bin/env bash
# Laptop: copy this project and the repo's gigi/ folder to the GPU box, in the
# repo's own layout, so follow-up parsers can load Gigi's src/common.py.
#
#   ./sync_to_box.sh <ssh-host> <ssh-port>
#
# Lands in /workspace/spar-model-hermeneutics/{ravi/week1-dyl-vs-oa/dyl-probe-repro,gigi}.
# Left behind: scratch/ (the Hub is the record), caches, and Gigi's .env files.
set -euo pipefail

host="${1:?usage: ./sync_to_box.sh <ssh-host> <ssh-port>}"
port="${2:?usage: ./sync_to_box.sh <ssh-host> <ssh-port>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REMOTE=/workspace/spar-model-hermeneutics
SSH=(ssh -i "$HOME/.ssh/vast_spar_ed25519" -p "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

tar -C "$REPO" -czf - \
  --exclude='ravi/week1-dyl-vs-oa/dyl-probe-repro/scratch' \
  --exclude='__pycache__' \
  --exclude='gigi/*/.env' \
  --exclude='gigi/*/data' \
  ravi/week1-dyl-vs-oa/dyl-probe-repro gigi \
  | "${SSH[@]}" "root@$host" "mkdir -p $REMOTE && tar -xzf - -C $REMOTE && cd $REMOTE \
      && echo files: \$(find . -type f | wc -l) \
      && (grep -rlI \$(printf '\r') ravi/week1-dyl-vs-oa/dyl-probe-repro || echo no-CRLF) \
      && (find gigi -name .env | grep . && echo 'GIGI .env COPIED' || echo no-gigi-env)"
