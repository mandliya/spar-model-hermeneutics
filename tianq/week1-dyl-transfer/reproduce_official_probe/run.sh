#!/usr/bin/env bash
# Known-answer test for the released DYL checkpoint.
#
# The city/country rollouts built here are the probe's own TRAINING
# distribution - the DYL paper fits the probe on exactly this construction
# (honest / lie-then-confess / lie-then-double-down system prompts over
# 'which country is this city in'). So a high score here says nothing about
# how good the probe is. It says our pipeline applies it correctly: right
# token, right layer, right sign. That is the only claim being made.
#
# It has to run before any transfer number is read. A misapplied probe scores
# chance everywhere, which is indistinguishable from a genuine 'does not
# generalise' result - and that is exactly what happened here until this
# check existed.
#
# Ravi's --train flag is deliberately NOT used. It fits a fresh probe on the
# same activations, which only tells you something when THIS check fails: if
# the released probe is at chance but a freshly fitted one separates cleanly,
# the fault is in how the checkpoint is applied rather than in extraction.
# Useful for debugging, redundant otherwise, and ~4 minutes of GPU either way:
#
#   cd ../../ravi/week1-dyl-vs-oa && PYTHONPATH=src python scripts/02_train_dyl_probe.py --train
#
#   ./reproduce_official_probe/run.sh
#
# No new code: Ravi already wrote and debugged this path, including the
# z-scoring fix, so re-implementing it would only create something to drift.
#
# His script writes into his own results/ directory, which the root .gitignore
# already excludes. That is deliberate rather than sloppy: `load_config` reads
# YAML values straight into the dataclass without coercing them to Path, so
# redirecting results_dir through a config file makes `cfg.results_dir / "..."`
# raise TypeError. Running it as-is and copying the output is the honest way
# to reuse it untouched.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$HERE")"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# No venv. This runs on a rented box that exists for one job, so deps go
# straight into the container's python:
#   pip install -r requirements.txt
[[ -f .env ]] || { echo "No .env here - copy .env.example and fill it in." >&2; exit 1; }
set -a
# shellcheck disable=SC1091
source .env
set +a

RAVI_DIR="$REPO_ROOT/ravi/week1-dyl-vs-oa"
DEST="$PROJECT_DIR/scratch/checkpoint"
mkdir -p "$DEST"

echo "=== downloading and unpacking the released checkpoint ==="
cd "$RAVI_DIR"
PYTHONPATH=src python scripts/02_train_dyl_probe.py --load-official

echo
echo "=== copying artifacts to $DEST ==="
for name in dyl_official_probe.json dyl_official_meta.json; do
  if [[ -f "$RAVI_DIR/results/$name" ]]; then
    cp "$RAVI_DIR/results/$name" "$DEST/$name"
    echo "  $name"
  else
    echo "  MISSING: $name" >&2
  fi
done

echo
echo "=== known-answer test: does the released checkpoint fire? ==="
python "$HERE/validate_official_probe.py"
