#!/usr/bin/env bash
# Launch the runs a stopped rolling launch never started, all at once (MAXC=0), so every run is live on Modal
# within minutes and the laptop can go: runs already started carry on without their launcher.
#   QWEN="env:n ..." GEMMA="env:n ..." MODAL_PROFILE=... bash gigi/3-forensics/scripts/modal_rest.sh <stage>
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
[ -n "${QWEN:-}" ] && MAXC=0 GAP=${GAP:-8} bash "$HERE/scripts/modal_full.sh" "$1" qwen3-next-80b $QWEN
[ -n "${GEMMA:-}" ] && MAXC=0 GAP=${GAP:-8} bash "$HERE/scripts/modal_full.sh" "$1" gemma-3-27b $GEMMA
echo "$(date '+%F %T') rest launched" >> "$HERE/results/logs/$1/launch.txt"
