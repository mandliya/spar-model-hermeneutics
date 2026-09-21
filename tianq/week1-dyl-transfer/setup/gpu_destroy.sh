#!/usr/bin/env bash
# Destroy the rented GPU instance and stop it billing.
#
#   ./setup/gpu_destroy.sh            # destroy whatever gpu_rent.sh recorded
#   ./setup/gpu_destroy.sh 12345678   # destroy a specific instance id
#
# The expensive failure here is not renting the wrong box, it is renting the
# right one and forgetting it: an 80GB card bills around the clock whether or
# not anything is using it.
#
# Two things this script learned the hard way, both of which produced a
# confident "Billing stopped" over an instance that was still running:
#
#   `vastai destroy instance` prompts for confirmation. With stdin not a TTY it
#   reads EOF and aborts — exit status looks fine, nothing is destroyed. Hence
#   the mandatory -y.
#
#   Success was printed unconditionally, and the recorded instance id was
#   deleted before anything checked. That threw away the only pointer to a box
#   that was still charging. Now the id is kept until the API confirms the
#   instance is gone, and a failed teardown exits non-zero and loudly.
#
# `vastai stop` is deliberately never used: it pauses GPU charges but keeps
# billing disk.
set -euo pipefail

STATE="$HOME/.vast_active_instance"

VENV_BIN="$HOME/.venvs/dyl-transfer/Scripts"
[[ -d "$VENV_BIN" ]] || VENV_BIN="$HOME/.venvs/dyl-transfer/bin"
VASTAI="$VENV_BIN/vastai"
[[ -x "$VASTAI" ]] || VASTAI="$VENV_BIN/vastai.exe"

export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

INSTANCE_ID="${1:-}"
if [[ -z "$INSTANCE_ID" && -f "$STATE" ]]; then
  INSTANCE_ID="$(tr -d '[:space:]' < "$STATE")"
fi

if [[ -z "$INSTANCE_ID" ]]; then
  echo "No instance id given and none recorded at $STATE."
  echo "Anything still running will be listed here:"
  "$VASTAI" show instances
  exit 1
fi

echo "Destroying instance $INSTANCE_ID ..."
"$VASTAI" destroy instance "$INSTANCE_ID" -y

echo "Verifying ..."
sleep 8

STILL_THERE="$("$VASTAI" show instances --raw 2>/dev/null | python -c "
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    print('unknown')
    sys.exit()
print('yes' if any(str(i.get('id')) == '$INSTANCE_ID' for i in data) else 'no')
")"

case "$STILL_THERE" in
  no)
    rm -f "$STATE"
    echo
    echo "Confirmed destroyed. Billing stopped."
    "$VASTAI" show instances
    ;;
  *)
    echo
    echo "TEARDOWN DID NOT TAKE EFFECT - instance $INSTANCE_ID may still be billing." >&2
    echo "The id is deliberately still recorded at $STATE." >&2
    echo "Destroy it by hand at https://cloud.vast.ai/instances/" >&2
    exit 1
    ;;
esac
