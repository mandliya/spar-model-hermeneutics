#!/usr/bin/env bash
# Laptop: copy this project and the repo's gigi/ folder to the GPU box, in the
# repo's own layout, so the code finds Gigi's files without any path settings.
#
#   ./sync_to_box.sh <ssh-host> <ssh-port>
#
# Lands in /workspace/spar-model-hermeneutics/{tianq/dyl-probe-repro,gigi}.
# Left behind: scratch/ (the Hub is the record), caches, and Gigi's .env files,
# which hold his own API key. This project's .env (the HF token) goes along.
# Run through logged.sh with DYL_SESSION_LOG pointing at the orchestration log.
set -euo pipefail

host="${1:?usage: ./sync_to_box.sh <ssh-host> <ssh-port>}"
port="${2:?usage: ./sync_to_box.sh <ssh-host> <ssh-port>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE=/workspace/spar-model-hermeneutics
SSH=(ssh -i "$HOME/.ssh/id_ed25519_vast" -p "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

tar -C "$REPO" -czf - \
  --exclude='tianq/dyl-probe-repro/scratch' \
  --exclude='__pycache__' \
  --exclude='gigi/*/.env' \
  --exclude='gigi/*/data' \
  tianq/dyl-probe-repro gigi \
  | "${SSH[@]}" "root@$host" "mkdir -p $REMOTE && tar -xzf - -C $REMOTE && cd $REMOTE \
      && echo files: \$(find . -type f | wc -l) \
      && (grep -rlI \$(printf '\r') tianq/dyl-probe-repro || echo no-CRLF) \
      && (find gigi -name .env | grep . && echo 'GIGI .env COPIED' || echo no-gigi-env)"
