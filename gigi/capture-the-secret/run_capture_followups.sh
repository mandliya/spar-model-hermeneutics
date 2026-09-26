#!/usr/bin/env bash
# capture-the-secret: gate, then the 12 follow-up questions, then upload.
#
# The gate is not optional. It refuses to start ~6000 generations if either
# arm is under 20 rows, because a self-report number with no usable control
# is what mask-factual and unanswerable both produced when nobody checked
# first.
#
#   nohup bash run_capture_followups.sh > results/capture_followups.log 2>&1 & disown

set -o pipefail
cd "$(dirname "$0")"
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }

say "=== GATE ==="
if python 02_analyse_capture_secret.py --gate; then
  say "--- gate passed ---"
else
  say "!!! GATE FAILED — not running the follow-up questions. The primary"
  say "!!! false-denial rate above still stands on its own."
  exit 2
fi

say "=== 12 FOLLOW-UP QUESTIONS ==="
if python 00_capture_secret_selfreport_benchmark.py; then
  say "--- follow-up questions OK ---"
else
  say "!!! follow-up run FAILED — partial rows kept, resumable"
  exit 1
fi

say "=== ANALYSIS ==="
python 02_analyse_capture_secret.py || say "!!! analysis failed (data safe)"

say "=== UPLOAD ACTIVATIONS ==="
bash upload_activations.sh capture_secret '*__capture__*' || say "!!! primary-run upload failed"

say "=== COMPLETE ==="
