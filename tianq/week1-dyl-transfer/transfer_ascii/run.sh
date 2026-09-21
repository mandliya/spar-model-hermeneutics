#!/usr/bin/env bash
# Apply the released DYL checkpoint, cold, to Gigi's ASCII-sandbagging rows.
#
#   ./transfer_ascii/run.sh                 # the real run, all 310 rows
#   ./transfer_ascii/run.sh --limit 8       # smoke test
#   ./transfer_ascii/run.sh --skip-extract  # rescore what is already stored
#
# Arguments pass straight through to eval_dyl_on_ant.py.
#
# Run ../reproduce_official_probe/run.sh FIRST. Nothing here is interpretable until the
# checkpoint has been shown to reproduce its published number: a probe applied
# at the wrong layer scores chance on this dataset too, and chance looks exactly
# like a real "does not generalise" finding. That mistake cost a GPU session.
#
# Safe to kill and rerun. Every finished row is written locally the moment it
# completes and synced to HuggingFace in batches, and a restart lists what is
# already there and picks up from the gap.
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

# The job outlives an SSH drop under tmux. Resume makes this belt-and-braces
# rather than load-bearing.
if [[ -z "${TMUX:-}" ]] && command -v tmux >/dev/null 2>&1; then
  echo "Not inside tmux. Consider: tmux new -s transfer" >&2
fi

python "$HERE/eval_dyl_on_ant.py" "$@"

echo
echo "Results are on HuggingFace. The local scratch/ copy is a cache, not a record."
echo "Nothing has been committed - git is yours."
